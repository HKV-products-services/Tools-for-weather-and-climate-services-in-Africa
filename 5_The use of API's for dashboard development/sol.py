
import datetime
import os

import altair as alt
import ipyleaflet
import openmeteo_requests
import pandas as pd
import requests_cache
import solara
from ipywidgets import HTML
from openmeteo_sdk.Aggregation import Aggregation
from openmeteo_sdk.Variable import Variable
from retry_requests import retry

import TAHMO

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Create a TAHMO API wrapper and set credentials
api = TAHMO.apiWrapper()

TAHMO_USER='T.Stolp@hkv.nl'
TAHMO_PASSWORD='X9R@zNX6dCeq45K'

api.setCredentials(TAHMO_USER, TAHMO_PASSWORD)

station_list = list(api.getStations())
station_data = {}

for station in station_list:
    station_data[station] = api.getStations()[station]

station_default = station_list[0]
center_default = (station_data[station_default]['location']['latitude'], station_data[station_default]['location']['longitude'])

# Define reactive variables for station data
station = solara.reactive(station_default)
center = solara.reactive(center_default)

icon_rain = ipyleaflet.AwesomeIcon(
    name='tint',
    marker_color='white',
    icon_color='blue',
    spin=False
)

def set_station(value):
    station.value = value
    center.value = (station_data[value]['location']['latitude'], station_data[value]['location']['longitude'])


# def on_click(val):


#     # popup_list = []
#     # for s in station_list:
#     #     location = (station_data[s]['location']['latitude'], station_data[s]['location']['longitude'])


#     #     popup_list.append(popup)


@solara.component
def StationSelect():
    """Solara component for a station selection dropdown."""
    solara.Select(label="station", values=station_list, value=station.value, on_value=set_station, style={"z-index": "10000"})
    
@solara.component
def View():
    """Solara component for displaying a map view with a marker for the selected station."""
    
    marker_list = []
    for s in station_list:
        location = (station_data[s]['location']['latitude'], station_data[s]['location']['longitude'])
        title = station_data[s]['location']['name']
        try:
            marker = ipyleaflet.Marker.element(icon=icon_rain, location=location, title=title, draggable=False)
            marker_list.append(marker)
        except Exception as e:
            print(e)

    ipyleaflet.Map.element(center=center.value,
                        zoom=6,
                        scroll_wheel_zoom=True, 
                        layers=[ipyleaflet.TileLayer.element(url=ipyleaflet.basemaps.OpenStreetMap.Mapnik.build_url())] + marker_list # + popup_list
                        )

def process_tahmo_precip_data(df):
    """Load the precipitation data from the TAHMO API and return a pandas dataframe"""
    df = df.reset_index().rename(columns={"Timestamp" : "date", "pr": "precipitation"})
    df['date'] = pd.to_datetime(df['date'])
    df.loc[:,'date'] = df['date'].dt.date
    df = df.groupby('date').sum().reset_index().dropna()
    df['date'] = pd.to_datetime(df['date'])
    return df
    
def request_precip_data(station, variables=['pr'], startDate='2023-01-01', endDate='2023-11-22'):
    """Request precipitation data from the TAHMO API and return a pandas dataframe."""
    df = api.getMeasurements(station, startDate=startDate, endDate=endDate, variables=variables)
    if df.empty:
        df = pd.DataFrame(columns=['date', 'precipitation'])
        return df
    else:
        df.index.name = 'Timestamp'
        df = df.reset_index()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.loc[:,'date'] = df['Timestamp'].dt.date
        df = df.drop(columns=['Timestamp']).groupby('date').max().reset_index().dropna()
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={"pr": "precipitation"})
        return df
    
def get_ecmwf_precipitation_ensemble(lon, lat):

	"""Retrieve the ECMWF precipitation forecast from the Open-Meteo API and return a JSON object"""
	
	url = "https://ensemble-api.open-meteo.com/v1/ensemble"
	
	params = {
		"latitude": lat,
		"longitude": lon,
		"forecast_days": 5,
		"past_days": 30,
		"hourly": "precipitation",
		"models": "ecmwf_ifs04"
	}
	responses = openmeteo.weather_api(url, params=params)

	response = responses[0]
	
	# Process hourly data
	hourly = response.Hourly()
	hourly_variables = list(map(lambda i: hourly.Variables(i), range(0, hourly.VariablesLength())))
	hourly_precipitation = filter(lambda x: x.Variable() == Variable.precipitation, hourly_variables)

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s"),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}
	# Process all members
	for variable in hourly_precipitation:
		member = variable.EnsembleMember()
		hourly_data[f"precipitation_member{member}"] = variable.ValuesAsNumpy()

	df = pd.DataFrame(data=hourly_data)
	return df

def process_ecmwf_ensemble_precip_data(df):
    """Load the precipitation data from the Open-Meteo API and return a pandas dataframe"""
    df = df.rename(columns={"date": "Timestamp"})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.loc[:,'date'] = df['Timestamp'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop(columns=['Timestamp'])
    df = df.groupby('date').sum().reset_index()
    
    return df

@solara.component
def Timeseries():
    
    """Solara component for a timeseries chart of precipitation."""	
    variables = ['pr']
    today = datetime.datetime.now()
    startDate = today - datetime.timedelta(days=30)
    df_tahmo = api.getMeasurements(station.value, startDate=startDate.strftime("%Y-%m-%d"), endDate=today.strftime("%Y-%m-%d"), variables=variables)
    df_tahmo.index.name = 'Timestamp'
    df_tahmo = process_tahmo_precip_data(df_tahmo)
    bar_tahmo =  alt.Chart(df_tahmo).mark_bar(opacity=0.75,).encode(x=alt.X("date:T"), y=alt.Y("precipitation:Q"), tooltip=['precipitation', 'date']).interactive()
    df_hourly = get_ecmwf_precipitation_ensemble(station_data[station.value]['location']['longitude'], station_data[station.value]['location']['latitude'])
    df_ecmwf_ensemble = process_ecmwf_ensemble_precip_data(df_hourly)
    ensemble_df = pd.DataFrame(data={'min' : df_ecmwf_ensemble.set_index('date').min(axis=1), 'max' : df_ecmwf_ensemble.set_index('date').max(axis=1), 'mean' : df_ecmwf_ensemble.set_index('date').mean(axis=1)}).reset_index()
    area_ecmwf = alt.Chart(ensemble_df).mark_area(opacity=0.25, color='orange').encode(x='date', y='min', y2='max').interactive()
    bar_ecmwf = alt.Chart(ensemble_df).mark_bar(opacity=0.75, color='orange').encode(x='date', y='mean', tooltip=['mean', 'date'])
    rule = alt.Chart(pd.DataFrame({'date': [today.strftime("%Y-%m-%d")], 'color': ['black']})).mark_rule().encode(x='date:T') 
    chart = area_ecmwf + rule + bar_tahmo + bar_ecmwf
    solara.display(chart.properties(width=1200, height=300, title=f"TAHMO station: {station_data[station.value]['location']['name']}").interactive())

@solara.component
def Page():
    """Solara component for a page with two cards: View and StationSelect."""
    
    with solara.AppBarTitle():
        solara.Text("TAHMO precipitation measurements vs. ECMWF ensemble forecast")
    solara.Info("Select a TAHMO station in the dropdown menu.")
    
    with solara.Column(style={"min-width": "500px", "height": "500px"}):
        with solara.Row():
            StationSelect()
        with solara.Columns([1, 2]):
            with solara.Card():
                View()
            with solara.Card():
                Timeseries()

Page()