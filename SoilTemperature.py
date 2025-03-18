import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date, datetime

# ---------------------------------------------------------------
# 1) Fetch Weather Data from the Open-Meteo Archive API
# ---------------------------------------------------------------
def fetch_weather_data(latitude, longitude, start_date, end_date, selected_vars):
    """
    Fetches hourly weather data from Open-Meteo for the given latitude, longitude,
    date range, and selected variables. Returns a pandas DataFrame.
    """
    # Setup caching and retry mechanism
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Map API variables to friendly names
    all_possible_vars = {
        "temperature_2m": "Air Temp (2m)",
        "soil_temperature_0_to_7cm": "Soil Temp (0-7cm)",
        "soil_temperature_7_to_28cm": "Soil Temp (7-28cm)",
        "soil_temperature_28_to_100cm": "Soil Temp (28-100cm)",
        "soil_temperature_100_to_255cm": "Soil Temp (100-255cm)"
    }

    # Convert friendly names back to API variable names
    chosen_vars = [k for k, v in all_possible_vars.items() if v in selected_vars]
    if not chosen_vars:
        return pd.DataFrame()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "hourly": chosen_vars
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    # Parse the response into a DataFrame
    response = responses[0]
    hourly = response.Hourly()
    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    data_dict = {"date": date_range}
    for i, var in enumerate(chosen_vars):
        data_dict[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data_dict)
    df.set_index("date", inplace=True)

    # If desired, you can convert from UTC to local time, e.g.:
    # df.index = df.index.tz_convert("America/New_York")
    return df


# ---------------------------------------------------------------
# 2) Plotting Functions
# ---------------------------------------------------------------
def plot_daily_timeseries(df):
    """
    Plots the daily time series for each variable in the DataFrame.
    """
    if df.empty:
        return None

    fig = go.Figure()
    rename_map = {
        "temperature_2m": "Air Temp (2m)",
        "soil_temperature_0_to_7cm": "Soil Temp (0-7cm)",
        "soil_temperature_7_to_28cm": "Soil Temp (7-28cm)",
        "soil_temperature_28_to_100cm": "Soil Temp (28-100cm)",
        "soil_temperature_100_to_255cm": "Soil Temp (100-255cm)"
    }
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines',
            name=rename_map.get(col, col),
            opacity=0.8
        ))
    fig.update_layout(
        title="Daily Temperature Time Series",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )
    return fig

def plot_monthly_averages(df):
    """
    Plots monthly average temperatures for each variable in the DataFrame.
    """
    if df.empty:
        return None

    monthly_avg = df.resample("MS").mean()
    fig = go.Figure()
    rename_map = {
        "temperature_2m": "Air Temp (2m)",
        "soil_temperature_0_to_7cm": "Soil Temp (0-7cm)",
        "soil_temperature_7_to_28cm": "Soil Temp (7-28cm)",
        "soil_temperature_28_to_100cm": "Soil Temp (28-100cm)",
        "soil_temperature_100_to_255cm": "Soil Temp (100-255cm)"
    }
    for col in monthly_avg.columns:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=rename_map.get(col, col)
        ))
    fig.update_layout(
        title="Monthly Average Temperatures",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )
    return fig

def plot_monthly_distribution_box(df):
    """
    Creates a box plot of monthly average temperatures for each variable.
    """
    if df.empty:
        return None

    monthly_avg = df.resample("MS").mean().reset_index()
    long_df = monthly_avg.melt(id_vars="date", var_name="Variable", value_name="Temperature")
    rename_map = {
        "temperature_2m": "Air Temp (2m)",
        "soil_temperature_0_to_7cm": "Soil Temp (0-7cm)",
        "soil_temperature_7_to_28cm": "Soil Temp (7-28cm)",
        "soil_temperature_28_to_100cm": "Soil Temp (28-100cm)",
        "soil_temperature_100_to_255cm": "Soil Temp (100-255cm)"
    }
    long_df["Variable"] = long_df["Variable"].map(rename_map)

    fig = px.box(
        long_df, x="Variable", y="Temperature",
        title="Monthly Distribution of Temperature by Depth",
        labels={"Variable": "Temperature Variable", "Temperature": "Temperature (°C)"}
    )
    return fig

def create_monthly_stats_table(df):
    """
    Returns a DataFrame with monthly mean, min, and max for each variable.
    """
    if df.empty:
        return pd.DataFrame()

    monthly_stats = df.resample("MS").agg(["mean", "min", "max"])
    monthly_stats.columns = [f"{var}_{stat}" for var, stat in monthly_stats.columns]
    monthly_stats.reset_index(inplace=True)
    monthly_stats.rename(columns={"date": "Month"}, inplace=True)
    return monthly_stats

def plot_correlation_heatmap(df):
    """
    Plots a correlation heatmap among the selected variables.
    """
    if df.empty:
        return None

    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    return fig


# ---------------------------------------------------------------
# 3) Streamlit App
# ---------------------------------------------------------------
def main():
    st.title("Weather Dashboard")

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    lat = st.sidebar.number_input("Latitude", value=41.5998779)
    lon = st.sidebar.number_input("Longitude", value=-85.3992676)
    start_date = st.sidebar.date_input("Start Date", date(2000, 1, 1))
    end_date = st.sidebar.date_input("End Date", date(2024, 12, 31))

    # Variable selection
    possible_vars = [
        "Air Temp (2m)",
        "Soil Temp (0-7cm)",
        "Soil Temp (7-28cm)",
        "Soil Temp (28-100cm)",
        "Soil Temp (100-255cm)"
    ]
    selected_vars = st.sidebar.multiselect(
        "Select Variables",
        options=possible_vars,
        default=possible_vars
    )

    # Fetch Data button
    if st.sidebar.button("Fetch Data"):
        if start_date >= end_date:
            st.error("Error: Start date must be before end date.")
            return

        st.info("Fetching data...")
        data = fetch_weather_data(lat, lon, start_date, end_date, selected_vars)
        if data.empty:
            st.error("No data fetched. Check variable selection or date range.")
            return

        st.success(f"Data fetched for {len(data)} hourly records.")
        st.write("Here is a preview of the first 10 rows:")
        st.dataframe(data.head(10))

        # Daily Time Series
        fig_daily = plot_daily_timeseries(data)
        if fig_daily:
            st.plotly_chart(fig_daily)

        # Monthly Averages
        fig_monthly = plot_monthly_averages(data)
        if fig_monthly:
            st.plotly_chart(fig_monthly)

        # Box Plot
        fig_box = plot_monthly_distribution_box(data)
        if fig_box:
            st.plotly_chart(fig_box)

        # Monthly Stats Table
        monthly_stats_df = create_monthly_stats_table(data)
        if not monthly_stats_df.empty:
            st.write("Detailed Monthly Stats (Mean, Min, Max):")
            st.dataframe(monthly_stats_df)

        # Correlation Heatmap
        if len(data.columns) > 1:
            fig_corr = plot_correlation_heatmap(data)
            if fig_corr:
                st.plotly_chart(fig_corr)

        # Save CSV button
        if st.sidebar.button("Save CSV"):
            data.to_csv('weather_data.csv')
            st.success("Data saved to 'weather_data.csv' in the current directory.")


# Run the app
if __name__ == "__main__":
    main()
