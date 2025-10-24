# -*- coding: utf-8 -*-
"""
Enhanced STEPS Nowcasting for RAINSAT (H40B/H60B)

- Mirrors the working call signature of your Nowcast class:
  uses nowcasts.get_method("steps") with timesteps, timestep, n_ens_members, etc.
- Uses dense Lucas–Kanade without the deprecated/unsupported 'dense' kwarg.
- Norain check via pysteps.utils.check_norain.check_norain.
- Fallback synthesis to guarantee >= 3 timesteps (interpolate/extrapolate).
- Safe CRS handling in export.
- THREDDS server integration for processed data and forecasts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import geopandas as gpd

# Register rioxarray accessor for .rio
import rioxarray  # noqa: F401

from shapely.geometry import box  # optional for callers
from pyproj import CRS

import pysteps
from pysteps import io, nowcasts, utils
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation
from pysteps.utils.check_norain import check_norain


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _ensure_min_timesteps(R: np.ndarray, time_coord, frequency_min: int) -> np.ndarray:
    """
    Ensure at least 3 time steps for motion estimation.

    - If exactly 2 steps and the gap ~ 2*frequency, linearly interpolate a middle frame.
    - Else if exactly 2 steps, linearly extrapolate a third frame forward.
    - If 1 step, replicate to reach 3 frames.

    Returns an array with shape (>=3, y, x), dtype float32, non-negative.
    """
    import pandas as pd

    def _finalize(arr):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        arr[arr < 0] = 0.0
        return arr

    R = np.asarray(R, dtype=np.float32)
    t = None
    if time_coord is not None and len(time_coord) == R.shape[0]:
        try:
            t = pd.to_datetime(time_coord)
        except Exception:
            t = None

    n = R.shape[0]
    if n >= 3:
        return _finalize(R)

    if n == 2:
        if t is not None:
            dt_min = abs((t[1] - t[0]).total_seconds()) / 60.0
            # Interpolate a missing middle frame if the gap ≈ 2*frequency
            if abs(dt_min - 2 * frequency_min) <= max(1.0, 0.2 * frequency_min):
                R_mid = 0.5 * (R[0] + R[1])
                R_new = np.stack([R[0], R_mid, R[1]], axis=0)
                return _finalize(R_new)
        # Otherwise extrapolate a third frame
        R3 = R[1] + (R[1] - R[0])
        R_new = np.stack([R[0], R[1], R3], axis=0)
        return _finalize(R_new)

    if n == 1:
        R_new = np.stack([R[0], R[0], R[0]], axis=0)
        return _finalize(R_new)

    # n == 0: upstream should catch; return sanitized zeros
    return _finalize(R)


def _crs_to_dict(crs_like) -> dict:
    """Normalize any CRS to a dict; include legacy 'proj' key if possible."""
    crs_obj = CRS.from_user_input(crs_like) if crs_like is not None else CRS.from_epsg(4326)
    d = crs_obj.to_dict()
    if "proj" not in d:
        try:
            d["proj"] = crs_obj.to_proj4_dict().get("proj", None)
        except Exception:
            d["proj"] = None
    return d


class EnhancedStepsNowcast:
    """
    Enhanced STEPS nowcasting optimized for H40B/H60B high-resolution data.
    Matches the working STEPS call style from your Nowcast implementation.
    """

    def __init__(self, settings: dict, data_source: str):
        self.settings = dict(settings)
        self.data_source = data_source.lower().strip()
        self.logger = LOGGER

        datafolder = Path(self.settings.get("datafolder", "./data"))
        self.outputfolder = datafolder / "nowcast"
        self.outputfolder.mkdir(parents=True, exist_ok=True)

        if self.data_source == "h40b":
            self.params = {
                "variable": "precip_intensity",
                "threshold": float(self.settings.get("threshold", 0.1)),  # mm/h
                "pixel_size": 2.0,  # km nominal
                "cascade_levels": 8,
                "buffer_distance": int(self.settings.get("buffer_distance", 500)),
            }
            
            
        elif self.data_source == "h60b":
            self.params = {
                "variable": "precip_intensity",
                "threshold": float(self.settings.get("threshold", 0.1)),
                "pixel_size": 3.0,
                "cascade_levels": 6,
                "buffer_distance": int(self.settings.get("buffer_distance", 500)),
            }
        else:
            raise ValueError(f"Unsupported data_source '{self.data_source}'")

        self.logger.info(
            "Enhanced HSAF configuration detected\n"
            f"  Source: {self.data_source.upper()}\n"
            f"  Variable: {self.params['variable']}\n"
            f"  Threshold: {self.params['threshold']} mm/h\n"
            f"  Pixel size: {self.params['pixel_size']} km"
        )

    # ---------------------------------------------------------------------
    # Data subsetting (inherited from Nowcast class)
    # ---------------------------------------------------------------------
    def subset_meteosat_data(self, country: str, datasets) -> tuple:
        """
        Subset the nowcasting input data for HSAF sources.
        
        Args:
            country (str): Current country
            datasets (Xarray): Datasets with training data

        Returns:
            tuple: (ds_country, gdf_country)
        """
        from shapely.geometry import box
        import geopandas as gpd
        
        buffer_distance = self.params["buffer_distance"]

        # Try to load country borders from database, with fallback
        try:
            from postgres import Postgres
            postgres = Postgres(self.settings)
            db_schema = self.settings["db_schema"]
            sql = f"SELECT * FROM {db_schema}.countries ORDER BY name"
            gdf_borders = postgres.get_geodata_by_query(query=sql)
            self.logger.info("Successfully loaded country boundaries from database")
        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
            self.logger.info("Using fallback country boundaries")
            gdf_borders = self._get_fallback_country_boundaries()

        # Select country
        gdf_country = gdf_borders[gdf_borders["name"] == country]
        if gdf_country.empty:
            self.logger.warning(f"Country '{country}' not found in boundaries, using fallback coordinates")
            gdf_country = self._get_fallback_country_boundary(country)
        else:
            gdf_country = gdf_country.dissolve(by="name")

        # HSAF data is already in WGS84
        target_crs = self.settings["crs_out"]  # WGS84
        gdf_country = gdf_country.to_crs(target_crs)
        
        # Convert buffer distance from meters to degrees (approximate)
        buffer_distance_deg = buffer_distance / 111000  # rough conversion
        
        # Clip grid with country shapefiles total bounds
        minx, miny, maxx, maxy = gdf_country.total_bounds
        
        # Apply the buffer in degrees
        new_minx = minx - buffer_distance_deg
        new_miny = miny - buffer_distance_deg
        new_maxx = maxx + buffer_distance_deg
        new_maxy = maxy + buffer_distance_deg

        # Create a rectangular polygon from the new extent
        buffered_extent_polygon = box(new_minx, new_miny, new_maxx, new_maxy)

        # Create a GeoDataFrame with the buffered extent polygon
        gdf_buffer = gpd.GeoDataFrame(
            [{"geometry": buffered_extent_polygon}], geometry="geometry"
        )
        gdf_buffer = gdf_buffer.set_crs(target_crs)

        ds_country = datasets.rio.clip(
            gdf_buffer.geometry.values,
            gdf_buffer.crs,
            drop=True,
            invert=False,
            all_touched=True,
        )

        return ds_country, gdf_country

    def _get_fallback_country_boundaries(self) -> gpd.GeoDataFrame:
        """
        Comprehensive African country boundaries when database is not available
        Covers all 54 African Union member states plus dependencies
        """
        from shapely.geometry import box
        import geopandas as gpd
        
        # Comprehensive African country bounding boxes (WGS84)
        # Format: (minx, miny, maxx, maxy) - (west, south, east, north)
        african_countries = {
            # North Africa
            "Algeria": (-8.7, 18.9, 12.0, 37.1),
            "Egypt": (24.7, 22.0, 36.9, 31.7),
            "Libya": (9.3, 19.5, 25.2, 33.2),
            "Morocco": (-17.1, 21.4, -1.0, 35.9),
            "Sudan": (21.8, 8.7, 38.6, 22.0),
            "Tunisia": (7.5, 30.2, 11.6, 37.5),
            "South Sudan": (24.1, 3.5, 35.9, 12.2),
            
            # West Africa
            "Benin": (0.8, 6.2, 3.8, 12.4),
            "Burkina Faso": (-5.5, 9.4, 2.4, 15.1),
            "Cape Verde": (-25.4, 14.8, -22.7, 17.2),
            "Ivory Coast": (-8.6, 4.2, -2.5, 10.7),
            "Gambia": (-16.8, 13.1, -13.8, 13.8),
            "Ghana": (-3.3, 4.7, 1.2, 11.2),
            "Guinea": (-15.1, 7.2, -7.6, 12.7),
            "Guinea-Bissau": (-16.7, 10.9, -13.6, 12.7),
            "Liberia": (-11.5, 4.4, -7.4, 8.6),
            "Mali": (-12.2, 10.1, 4.3, 25.0),
            "Mauritania": (-17.1, 14.7, -4.8, 27.3),
            "Niger": (0.2, 11.7, 16.0, 23.5),
            "Nigeria": (2.7, 4.3, 14.7, 13.9),
            "Senegal": (-17.5, 12.3, -11.3, 16.7),
            "Sierra Leone": (-13.3, 6.9, -10.3, 10.0),
            "Togo": (-0.1, 6.1, 1.8, 11.1),
            
            # Central Africa
            "Cameroon": (8.5, 1.7, 16.2, 13.1),
            "Central African Republic": (14.4, 2.2, 27.5, 11.0),
            "Chad": (13.5, 7.4, 24.0, 23.4),
            "Democratic Republic of the Congo": (12.2, -13.5, 31.3, 5.4),
            "Republic of the Congo": (11.1, -5.0, 18.6, 3.7),
            "Equatorial Guinea": (9.3, 0.9, 11.3, 2.3),
            "Gabon": (8.7, -4.0, 14.5, 2.3),
            "Sao Tome and Principe": (6.4, -0.0, 7.5, 1.7),
            
            # East Africa
            "Burundi": (29.0, -4.5, 30.8, -2.3),
            "Comoros": (43.2, -12.4, 44.5, -11.4),
            "Djibouti": (41.8, 10.9, 43.4, 12.7),
            "Eritrea": (36.4, 12.4, 43.1, 18.0),
            "Ethiopia": (33.0, 3.4, 48.0, 14.9),
            "Kenya": (33.9, -4.7, 41.9, 5.5),
            "Madagascar": (43.2, -25.6, 50.5, -11.9),
            "Malawi": (32.7, -17.1, 35.9, -9.4),
            "Mauritius": (56.5, -20.5, 63.5, -10.3),
            "Mozambique": (30.2, -26.9, 40.8, -10.5),
            "Rwanda": (28.9, -2.8, 30.9, -1.0),
            "Seychelles": (45.9, -10.4, 56.3, -3.7),
            "Somalia": (40.9, -1.7, 51.4, 12.0),
            "Tanzania": (29.3, -11.7, 40.4, -1.0),
            "Uganda": (29.6, -1.5, 35.0, 4.2),
            "Zambia": (21.9, -18.1, 33.7, -8.2),
            "Zimbabwe": (25.2, -22.4, 33.1, -15.6),
            
            # Southern Africa
            "Angola": (11.7, -18.0, 24.1, -4.4),
            "Botswana": (19.9, -26.9, 29.4, -17.8),
            "Eswatini": (30.8, -27.3, 32.1, -25.7),
            "Lesotho": (27.0, -30.7, 29.5, -28.6),
            "Namibia": (11.7, -29.0, 25.3, -16.9),
            "South Africa": (16.5, -47.1, 32.9, -22.1),
            
            # Additional territories and dependencies
            "Western Sahara": (-17.1, 20.8, -8.7, 27.7),
            "Mayotte": (45.0, -13.0, 45.3, -12.6),
            "Reunion": (55.2, -21.4, 55.8, -20.9),
            "Saint Helena": (-5.8, -16.0, -5.6, -15.9),
        }
        
        # Create GeoDataFrame with country boundaries
        geometries = []
        names = []
        
        for name, bounds in african_countries.items():
            minx, miny, maxx, maxy = bounds
            geometry = box(minx, miny, maxx, maxy)
            geometries.append(geometry)
            names.append(name)
        
        gdf = gpd.GeoDataFrame({
            'name': names,
            'geometry': geometries
        }, crs="EPSG:4326")
        
        self.logger.info(f"Loaded {len(african_countries)} African country boundaries")
        return gdf

    def _get_fallback_country_boundary(self, country: str) -> gpd.GeoDataFrame:
        """
        Get a single African country boundary as fallback
        """
        from shapely.geometry import box
        import geopandas as gpd
        
        # Get all African countries
        all_countries = self._get_fallback_country_boundaries()
        
        # Try to find the requested country
        country_match = all_countries[all_countries["name"].str.lower() == country.lower()]
        
        if not country_match.empty:
            return country_match
        
        # If exact match fails, try partial matching
        partial_match = all_countries[all_countries["name"].str.contains(country, case=False, na=False)]
        
        if not partial_match.empty:
            self.logger.info(f"Using partial match for '{country}': {partial_match.iloc[0]['name']}")
            return partial_match.iloc[0:1]  # Return first match as GeoDataFrame
        
        # Default fallback to center of Africa if country not found
        self.logger.warning(f"Country '{country}' not found, using default African bounds")
        default_bounds = (10.0, -10.0, 30.0, 10.0)  # Central Africa region
        
        minx, miny, maxx, maxy = default_bounds
        geometry = box(minx, miny, maxx, maxy)
        
        gdf = gpd.GeoDataFrame({
            'name': [country],
            'geometry': [geometry]
        }, crs="EPSG:4326")
        
        return gdf

    # ---------------------------------------------------------------------
    # Core nowcast (mirrors your working STEPS call semantics)
    # ---------------------------------------------------------------------
    def nowcast_steps_pysteps(self, datasets: xr.Dataset):
        """
        Run enhanced STEPS on an xarray Dataset (already subset to region).
        Returns:
            precip_forecast (np.ndarray or xarray-like), metadata (dict)
        """
        self.logger.info(f"Running enhanced STEPS nowcasting with {self.data_source.upper()}")

        n_ens_members = int(self.settings.get("ensemble", 1))
        transform_method = str(self.settings.get("transform", "dB"))
        frequency = int(self.settings.get("frequency", 15))
        n_lead_times = int(self.settings.get("n_lead_times", 6))
        max_workers = int(self.settings.get("max_workers", 2))
        zerovalue = float(self.settings.get("zerovalue", -15.0))

        variable_name = self.params["variable"]
        threshold = float(self.params["threshold"])
        kmperpixel = float(self.params["pixel_size"])
        n_cascade_levels = int(self.params["cascade_levels"])
        buffer_distance = int(self.params["buffer_distance"])

        if variable_name not in datasets:
            raise KeyError(f"Expected variable '{variable_name}' not found in dataset")

        precip_da = datasets[variable_name]

        # xarray -> numpy (t, y, x) + time coordinate
        if "time" in precip_da.dims:
            precip = precip_da.transpose("time", "y", "x").values.astype(np.float32)
            time_coord = precip_da["time"].values
        else:
            precip = precip_da.transpose("y", "x").values.astype(np.float32)[None, ...]
            time_coord = None

        # Clean and ensure 3 frames
        precip = np.nan_to_num(precip, nan=0.0, posinf=0.0, neginf=0.0)
        precip[precip < 0] = 0.0
        precip = _ensure_min_timesteps(precip, time_coord, frequency)
        
        # Build metadata (pysteps exporter needs x1..y2, yorigin, pixel sizes)
        precip_da = datasets[variable_name]
        x = np.asarray(precip_da["x"].values, dtype=float)
        y = np.asarray(precip_da["y"].values, dtype=float)

        # Determine y origin and extents
        if y.size >= 2 and y[0] < y[-1]:
            yorigin = "lower"
            y1, y2 = float(y.min()), float(y.max())
        else:
            # y decreasing -> origin upper
            yorigin = "upper"
            y1, y2 = float(y.max()), float(y.min())

        x1, x2 = float(x.min()), float(x.max())

        # Pixel sizes from coords (deg). Keep kmperpixel from settings.
        xpixelsize = float(abs(x[1] - x[0])) if x.size >= 2 else np.nan
        ypixelsize = float(abs(y[1] - y[0])) if y.size >= 2 else np.nan

        shape_yx = (precip.shape[1], precip.shape[2])

        metadata = {
            "accutime": frequency,
            "cartesian_unit": "km",
            "institution": "HKV services",
            "projection": "+proj=longlat +datum=WGS84 +no_defs",
            "threshold": threshold,          # mm/h
            "precip_thr": threshold,
            "norain_thr": float(self.settings.get("norain_thr", 0.005)),
            "unit": "mm/h",
            "xpixelsize": xpixelsize,        # degrees (ok for writer; kmperpixel is used for physics)
            "ypixelsize": ypixelsize,        # degrees
            "kmperpixel": kmperpixel,        # km (drives STEPS scales)
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
            "yorigin": yorigin,
            "zerovalue": float(self.settings.get("zerovalue", -15.0)),
            "transform": None,               # will be set by transformer below
            "n_lead_times": n_lead_times,
            "ensemble": n_ens_members,
            "num_workers": int(self.settings.get("max_workers", 2)),
            "shape": shape_yx,               # (y, x)
        }

        # Norain check
        if check_norain(precip, precip_thr=metadata["threshold"], norain_thr=metadata["norain_thr"]):
            self.logger.warning(
                "Input rainfall not above threshold; output will be zeros for all timesteps."
            )

            # Init streaming exporter
            fc_exporter = self._initialize_exporter_stream(
                metadata, shape_yx, self._default_startdate(datasets)
            )

            H, W = shape_yx
            T = metadata["n_lead_times"]
            E = metadata["ensemble"]

            if E == 1:
                # Return a (T, Y, X) cube so downstream export works
                zero_cube = np.zeros((T, H, W), dtype=np.float32)

                # Also stream per-timestep 2-D fields (Y, X)
                for t in range(T):
                    io.exporters.export_forecast_dataset(zero_cube[t], fc_exporter)

                io.exporters.close_forecast_files(fc_exporter)
                return zero_cube, metadata

            else:
                # Multi-ensemble: return (E, T, Y, X) and stream (E, Y, X) per step
                zero_cube = np.zeros((E, T, H, W), dtype=np.float32)

                for t in range(T):
                    io.exporters.export_forecast_dataset(zero_cube[:, t, :, :], fc_exporter)

                io.exporters.close_forecast_files(fc_exporter)
                return zero_cube, metadata

        # Transform (dB etc.) using pysteps utils to align with your script
        transformer = utils.get_method(transform_method)
        precip_t, metadata = transformer(
            precip,
            metadata,
            threshold=metadata["threshold"],
            zerovalue=metadata["zerovalue"],
            inverse=False,
        )
        precip_t[~np.isfinite(precip_t)] = metadata["zerovalue"]

        # Motion field (dense Lucas–Kanade); pass buffer via fd_kwargs
        velocity = dense_lucaskanade(precip_t, fd_kwargs={"buffer_mask": buffer_distance})

        # Exporter setup (streaming incremental output)
        fc_exporter = self._initialize_exporter_stream(metadata, shape_yx, self._default_startdate(datasets))

        def exporter(precip_transformed):
            # If ensemble==1, exporter expects 2D (y, x). Squeeze any leading size-1 axis.
            if precip_transformed.ndim == 3 and precip_transformed.shape[0] == 1:
                precip_transformed = precip_transformed[0]

            precip_forecast, _ = transformer(
                precip_transformed,
                metadata=metadata,
                threshold=metadata["threshold"],
                inverse=True,
            )

            if precip_forecast.ndim == 3 and precip_forecast.shape[0] == 1:
                precip_forecast = precip_forecast[0]

            io.exporters.export_forecast_dataset(precip_forecast, fc_exporter)

        # STEPS call
        steps = nowcasts.get_method("steps")
        steps(
            precip=precip_t[-3:, :, :],        # last 3 frames
            velocity=velocity,
            timesteps=metadata["n_lead_times"],
            timestep=metadata["accutime"],
            n_ens_members=n_ens_members,
            n_cascade_levels=n_cascade_levels,
            precip_thr=metadata["threshold"],
            kmperpixel=metadata["kmperpixel"],
            noise_method="nonparametric",
            vel_pert_method=None,
            conditional=False,
            mask_method="incremental",
            probmatching_method="cdf",
            seed=None,
            num_workers=metadata["num_workers"],
            callback=exporter,
        )

        io.exporters.close_forecast_files(fc_exporter)


        return None, metadata

    # ---------------------------------------------------------------------
    # Export helpers (aligned with your streaming approach)
    # ---------------------------------------------------------------------
    def _default_startdate(self, datasets: xr.Dataset):
        """Pick start date from dataset time if available; else now (UTC)."""
        try:
            if "time" in datasets:
                t = datasets["time"].values
                if np.ndim(t) > 0 and t.size > 0:
                    return np.datetime64(t[-1]).astype("datetime64[s]").astype(object)
        except Exception:
            pass
        return datetime.now(timezone.utc)

    def _initialize_exporter_stream(self, metadata: dict, shape_yx, start_date):
        """Initialize a streaming NetCDF exporter (incremental='timestep')."""
        outname = f"pysteps_{self.data_source}_latest.nc"
        netcdf_nowcast = self.outputfolder.joinpath(outname)
        fc_exporter = io.exporters.initialize_forecast_exporter_netcdf(
            outpath=netcdf_nowcast.resolve().parent,
            outfnprefix=netcdf_nowcast.stem,
            startdate=start_date,
            timestep=metadata["accutime"],
            n_timesteps=metadata["n_lead_times"],
            shape=shape_yx,
            metadata=metadata,
            n_ens_members=metadata["ensemble"],
            incremental="timestep",
            kwargs={"institution": metadata.get("institution", "")},
        )
        return fc_exporter

    # ---------------------------------------------------------------------
    # Enhanced export with THREDDS server integration
    # ---------------------------------------------------------------------
    def export_nowcast_to_netcdf(
        self,
        country: str,
        nowcasting_arrays: np.ndarray | None,
        date_start,
        metadata: dict,
        reproject: bool = True,
        data_source: str | None = None,
    ):
        """
        Enhanced export with THREDDS server integration
        Saves both processed files and forecasts to THREDDS server
        """
        ds_label = (data_source or self.data_source).lower()
        self.logger.info(f"Exporting {ds_label.upper()} nowcast for {country}")

        country_safe = country.lower().replace(" ", "_")
        out_stem = f"pysteps_{ds_label}_{country_safe}"
        netcdf_nowcast = self.outputfolder / f"{out_stem}.nc"

        # Get THREDDS path from settings
        thredds_path = self.settings.get("threddsdata", "/thredds_data/update_rainsat/")
        thredds_dir = Path(thredds_path)
        thredds_dir.mkdir(parents=True, exist_ok=True)

        # If no arrays provided (streamed already), copy the streaming output to THREDDS
        if nowcasting_arrays is None:
            self.logger.info("Copying streaming forecast output to THREDDS server...")
            
            # Copy the latest forecast to THREDDS
            latest_forecast = self.outputfolder / f"pysteps_{self.data_source}_latest.nc"
            if latest_forecast.exists():
                thredds_forecast = thredds_dir / f"forecast_{ds_label}_{country_safe}.nc"
                
                # Load, enhance, and save to THREDDS
                try:
                    import shutil
                    ds = xr.open_dataset(latest_forecast)
                    
                    # Enhance with country-specific metadata
                    ds.attrs.update({
                        "title": f"RAINSAT {ds_label.upper()} Nowcast - {country}",
                        "institution": "HKV / Project RAINSAT",
                        "source": f"HSAF {ds_label.upper()}",
                        "country": country,
                        "forecast_reference_time": date_start.isoformat() if date_start else "",
                        "history": f"Created {datetime.now(timezone.utc).isoformat()}",
                        "references": "https://pysteps.readthedocs.io/",
                        "Conventions": "CF-1.8",
                        "projection": "EPSG:4326",
                        "data_source": f"hsaf-{ds_label}"
                    })
                    
                    # Save to THREDDS
                    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
                    ds.to_netcdf(thredds_forecast, mode="w", encoding=encoding)
                    ds.close()
                    
                    self.logger.info(f"Forecast saved to THREDDS: {thredds_forecast}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to copy forecast to THREDDS: {e}")
                    
            return

        # Standard export path for in-memory arrays
        n_lead_times = int(metadata.get("n_lead_times", self.settings.get("n_lead_times", 6)))
        frequency = int(metadata.get("accutime", self.settings.get("frequency", 15)))
        shape_yx = metadata.get("shape")

        exporter = io.exporters.initialize_forecast_exporter_netcdf(
            outpath=netcdf_nowcast.resolve().parent,
            outfnprefix=netcdf_nowcast.stem,
            startdate=date_start,
            timestep=frequency,
            n_timesteps=n_lead_times,
            shape=shape_yx,
            metadata=metadata,
            n_ens_members=int(metadata.get("ensemble", 1)),
            incremental=None,
        )
        io.exporters.export_forecast_dataset(nowcasting_arrays, exporter)
        io.exporters.close_forecast_files(exporter)

        if reproject:
            self._reproject_and_save_to_thredds(netcdf_nowcast, country, ds_label, date_start)

    def _reproject_and_save_to_thredds(self, netcdf_nowcast: Path, country: str, ds_label: str, date_start):
        """Reproject and save to THREDDS server"""
        country_safe = country.lower().replace(" ", "_")
        
        # Get THREDDS path
        thredds_path = self.settings.get("threddsdata", "/thredds_data/update_rainsat/")
        thredds_dir = Path(thredds_path)
        thredds_dir.mkdir(parents=True, exist_ok=True)
        
        thredds_file = thredds_dir / f"forecast_{ds_label}_{country_safe}.nc"

        try:
            ds = xr.open_dataset(netcdf_nowcast.resolve())

            # Attach CRS if missing; in this workflow EPSG:4326 is typical
            if not ds.rio.crs:
                ds = ds.rio.write_crs("EPSG:4326", inplace=False)

            # Drop 2D lat/lon if present
            ds = ds.drop_vars(["lat", "lon"], errors="ignore")

            # Enhanced global attributes for THREDDS
            ds.attrs.update({
                "title": f"RAINSAT {ds_label.upper()} Nowcast - {country}",
                "institution": "HKV / Project RAINSAT",
                "source": f"HSAF {ds_label.upper()}",
                "country": country,
                "forecast_reference_time": date_start.isoformat() if date_start else "",
                "history": f"Created {datetime.now(timezone.utc).isoformat()}",
                "references": "https://pysteps.readthedocs.io/",
                "Conventions": "CF-1.8",
                "projection": "EPSG:4326",
                "data_source": f"hsaf-{ds_label}",
                "forecast_duration": f"{self.settings.get('n_lead_times', 6) * self.settings.get('frequency', 15)} minutes",
                "forecast_interval": f"{self.settings.get('frequency', 15)} minutes"
            })

            # Enhanced coordinate attributes
            if "y" in ds:
                ds["y"].attrs.update({
                    "axis": "Y",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                })
            if "x" in ds:
                ds["x"].attrs.update({
                    "axis": "X", 
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                })

            # Enhanced variable attributes
            if "precip_intensity" in ds:
                ds["precip_intensity"].attrs.update({
                    "long_name": "Precipitation Intensity Forecast",
                    "standard_name": "precipitation_flux",
                    "units": "mm/hr",
                    "data_source": f"hsaf-{ds_label}",
                    "country": country,
                    "forecast_type": "nowcast"
                })

            # Save with compression
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            ds.to_netcdf(thredds_file, mode="w", encoding=encoding)
            ds.close()
            
            self.logger.info(f"Forecast saved to THREDDS server: {thredds_file}")

        except Exception as e:
            self.logger.error(f"Failed to save forecast to THREDDS: {e}")
            raise

    def export_processed_data_to_thredds(self, processed_files: list, source_label: str):
            """
            Export processed HSAF data to THREDDS server
            
            Args:
                processed_files: List of processed file paths
                source_label: Source identifier (h40b/h60b)
            """
            thredds_path = self.settings.get("threddsdata", "/thredds_data/update_rainsat/")
            thredds_dir = Path(thredds_path)
            thredds_processed_dir = thredds_dir / "processed" / source_label
            thredds_processed_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Exporting {len(processed_files)} processed {source_label.upper()} files to THREDDS")
            
            for file_path in processed_files:
                try:
                    file_path = Path(file_path)
                    
                    # Use clean RAINSAT filename directly
                    if file_path.name.startswith("RAINSAT_"):
                        # Already in correct format, use as-is
                        thredds_filename = file_path.name
                    else:
                        # Fallback for any legacy files - extract RAINSAT part if present
                        import re
                        rainsat_match = re.search(r'(RAINSAT_\d{8}T\d{6}\.nc)', file_path.name)
                        if rainsat_match:
                            thredds_filename = rainsat_match.group(1)
                        else:
                            # Last resort: use original filename
                            thredds_filename = file_path.name
                    
                    thredds_file = thredds_processed_dir / thredds_filename
                    
                    # Copy processed file to THREDDS with enhanced metadata
                    ds = xr.open_dataset(file_path)
                    
                    # Add THREDDS-specific attributes
                    ds.attrs.update({
                        "thredds_catalog": "RAINSAT Processed Data",
                        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
                        "original_filename": file_path.name,
                        "data_quality": "operational",
                        "update_frequency": "10-15 minutes"
                    })
                    
                    # Save with compression
                    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
                    ds.to_netcdf(thredds_file, encoding=encoding)
                    ds.close()
                    
                    self.logger.info(f"Processed file exported to THREDDS: {thredds_filename}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to export {file_path} to THREDDS: {e}")
                    continue
            
            # Create latest symlink for easy access
            try:
                if processed_files:
                    latest_file = thredds_processed_dir / f"{source_label}_latest.nc" 
                    if latest_file.is_symlink():
                        latest_file.unlink()
                    
                    # Point to most recent RAINSAT file
                    rainsat_files = list(thredds_processed_dir.glob("RAINSAT_*.nc"))
                    if rainsat_files:
                        # Sort by timestamp in filename
                        most_recent = max(rainsat_files, key=lambda x: x.name)
                        latest_file.symlink_to(most_recent.name)
                        self.logger.info(f"Updated latest symlink: {latest_file}")
                    else:
                        # Fallback to any processed files
                        most_recent = max(thredds_processed_dir.glob(f"{source_label}_processed_*.nc"), default=None)
                        if most_recent:
                            latest_file.symlink_to(most_recent.name)
                            self.logger.info(f"Updated latest symlink: {latest_file}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to create latest symlink: {e}")