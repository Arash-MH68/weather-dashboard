import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date

# ===============================================================
# CONFIGURATION & GLOBAL IMPROVEMENTS
# ===============================================================

# Set page configuration with a professional title and logo
st.set_page_config(
    page_title="Terracon Weather Dashboard",
    page_icon="🌎",
    layout="centered"
)

# Terracon Logo (improvement: brand identity)
st.image("https://images.app.goo.gl/PfMR7rrkgT5YC5yQ8", width=200)

# ---------------------------------------------------------------
# Data Caching Function (improved with error handling and docstring)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_weather_data(latitude, longitude, start_date, end_date, selected_vars):
    """
    Fetches hourly weather data from Open-Meteo for the given location and date range.
    Includes both soil temperature and soil moisture data based on the selected variables.
    
    Parameters:
        latitude (float): Latitude of location.
        longitude (float): Longitude of location.
        start_date (date): Start date for data.
        end_date (date): End date for data.
        selected_vars (list): List of API variable names to fetch.
    
    Returns:
        pd.DataFrame: DataFrame containing hourly records with datetime index.
    """
    # Setup caching and retry mechanism for reliability
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # All possible API variables with friendly names
    all_possible_vars = {
        "temperature_2m": "Air Temp (2m)",
        "soil_temperature_0_to_7cm": "Soil Temp (0-7cm)",
        "soil_temperature_7_to_28cm": "Soil Temp (7-28cm)",
        "soil_temperature_28_to_100cm": "Soil Temp (28-100cm)",
        "soil_temperature_100_to_255cm": "Soil Temp (100-255cm)",
        "soil_moisture_0_to_7cm": "Soil Moisture (0-7cm)",
        "soil_moisture_7_to_28cm": "Soil Moisture (7-28cm)",
        "soil_moisture_28_to_100cm": "Soil Moisture (28-100cm)",
        "soil_moisture_100_to_255cm": "Soil Moisture (100-255cm)"
    }

    # Convert selected friendly names to API variable names
    chosen_vars = [api_var for api_var, friendly in all_possible_vars.items() if friendly in selected_vars]
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

    response = responses[0]
    hourly = response.Hourly()

    # Create time range from API parameters
    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Construct the data dictionary
    data_dict = {"date": date_range}
    for i, var in enumerate(chosen_vars):
        data_dict[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data_dict)
    df.set_index("date", inplace=True)
    return df

# ===============================================================
# PLOTTING FUNCTIONS
# ===============================================================

def plot_daily_timeseries(df):
    """
    Plot daily time series for all variables.
    """
    if df.empty:
        return None

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines',
            name=col,
            opacity=0.8
        ))
    fig.update_layout(
        title="Daily Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_monthly_averages(df):
    """
    Plot monthly averages for all variables.
    """
    if df.empty:
        return None

    monthly_avg = df.resample("MS").mean()
    fig = go.Figure()
    for col in monthly_avg.columns:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=col
        ))
    fig.update_layout(
        title="Monthly Averages",
        xaxis_title="Month",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_soil_temperature_profile_monthly(df):
    """
    Plot monthly average soil temperature profiles (by depth).
    """
    if df.empty:
        return None

    # Select only soil temperature variables
    soil_temp_vars = [c for c in df.columns if "soil_temperature" in c]
    if not soil_temp_vars:
        return None

    monthly_avg = df.resample("MS").mean()
    fig = go.Figure()
    for col in soil_temp_vars:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=col
        ))
    fig.update_layout(
        title="Monthly Average Soil Temperature Profiles",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_soil_moisture_profile_monthly(df):
    """
    Plot monthly average soil moisture profiles (by depth).
    """
    if df.empty:
        return None

    # Select only soil moisture variables
    soil_moist_vars = [c for c in df.columns if "soil_moisture" in c]
    if not soil_moist_vars:
        return None

    monthly_avg = df.resample("MS").mean()
    fig = go.Figure()
    for col in soil_moist_vars:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=col
        ))
    fig.update_layout(
        title="Monthly Average Soil Moisture Profiles",
        xaxis_title="Month",
        yaxis_title="Soil Moisture (m³/m³)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_vertical_profile(df, month_year, profile_type):
    """
    Plot a vertical profile (average vs. depth) for soil temperature or moisture 
    for a selected month-year.
    
    Parameters:
        df (pd.DataFrame): The complete data.
        month_year (str): Month-year string in the format "YYYY-MM".
        profile_type (str): Either "Temperature" or "Moisture".
    Returns:
        Plotly Figure with depth on y-axis and the average value on x-axis.
    """
    if df.empty:
        return None

    # Filter data for the selected month-year
    filtered = df[df.index.strftime("%Y-%m") == month_year]
    if filtered.empty:
        st.warning("No data available for the selected month–year.")
        return None

    # Define depth mapping (cm) for soil variables only
    depths_temperature = {
        "soil_temperature_0_to_7cm": 3.5,
        "soil_temperature_7_to_28cm": 17.5,
        "soil_temperature_28_to_100cm": 64,
        "soil_temperature_100_to_255cm": 177.5
    }
    depths_moisture = {
        "soil_moisture_0_to_7cm": 3.5,
        "soil_moisture_7_to_28cm": 17.5,
        "soil_moisture_28_to_100cm": 64,
        "soil_moisture_100_to_255cm": 177.5
    }

    if profile_type == "Temperature":
        vars_depth = depths_temperature
    else:
        vars_depth = depths_moisture

    avg_values = {}
    for var, depth in vars_depth.items():
        if var in filtered.columns:
            avg_values[depth] = filtered[var].mean()

    if not avg_values:
        st.warning(f"No {profile_type.lower()} data available for the selected month–year.")
        return None

    # Sort by depth (ascending)
    depths = sorted(avg_values.keys())
    values = [avg_values[d] for d in depths]

    # Create vertical profile plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=depths,
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=10),
        name=f"{profile_type} Profile"
    ))
    fig.update_layout(
        title=f"{profile_type} Vertical Profile for {month_year}",
        xaxis_title=f"{profile_type} Value",
        yaxis_title="Depth (cm)",
        yaxis=dict(autorange="reversed"),  # Depth increasing downward
        template="plotly_white",
        hovermode="y"
    )
    return fig

# ===============================================================
# STREAMLIT APP LAYOUT & INTERACTIVITY
# ===============================================================
def main():
    # Developer Credit and Title (with logo already shown above)
    st.markdown("<h2 style='text-align: center;'>Terracon Weather Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar About Section (improved description and usage instructions)
    with st.sidebar.expander("ℹ️ About this App", expanded=False):
        st.write(
            """
            This professional weather dashboard fetches historical weather data from Open-Meteo.
            Select your desired location and date range, then choose between air/soil temperature and soil moisture.
            Visualizations include daily time series, monthly averages, and interactive soil vertical profiles.
            """
        )

    # Sidebar Inputs (separated into Temperature and Moisture groups)
    st.sidebar.header("Input Parameters")
    lat = st.sidebar.number_input("Latitude", value=41.5998779, help="Enter the latitude (e.g., 41.5998779)")
    lon = st.sidebar.number_input("Longitude", value=-85.3992676, help="Enter the longitude (e.g., -85.3992676)")
    start_date = st.sidebar.date_input("Start Date", date(2010, 1, 1), help="Select the start date")
    end_date = st.sidebar.date_input("End Date", date(2024, 12, 31), help="Select the end date")

    # Separate variable selection for Temperature and Moisture
    st.sidebar.subheader("Temperature Variables")
    temp_options = ["Air Temp (2m)", "Soil Temp (0-7cm)", "Soil Temp (7-28cm)",
                    "Soil Temp (28-100cm)", "Soil Temp (100-255cm)"]
    selected_temp = st.sidebar.multiselect("Select Temperature Variables", options=temp_options, default=temp_options)

    st.sidebar.subheader("Soil Moisture Variables")
    moist_options = ["Soil Moisture (0-7cm)", "Soil Moisture (7-28cm)",
                     "Soil Moisture (28-100cm)", "Soil Moisture (100-255cm)"]
    selected_moist = st.sidebar.multiselect("Select Moisture Variables", options=moist_options, default=moist_options)

    # Combine the two selections for API query
    selected_vars = selected_temp + selected_moist

    # Plot Selection Options (improved interactivity)
    st.sidebar.markdown("#### Choose Plots to Display:")
    show_daily = st.sidebar.checkbox("Daily Time Series", value=True)
    show_monthly = st.sidebar.checkbox("Monthly Averages", value=True)
    show_soil_temp = st.sidebar.checkbox("Soil Temperature Profiles (Monthly)", value=True)
    show_soil_moist = st.sidebar.checkbox("Soil Moisture Profiles (Monthly)", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Vertical Profile Explorer")
    profile_type = st.sidebar.radio("Select Profile Type", options=["Temperature", "Moisture"])
    month_options = None  # Will be populated after data is fetched
    selected_month_year = st.sidebar.selectbox("Select Month–Year", options=[""], index=0)

    # ---------------------------------------------------------------
    # Data Fetch and Display
    # ---------------------------------------------------------------
    if st.sidebar.button("Fetch Data"):
        if start_date > end_date:
            st.error("Error: Start date must be before or equal to End date.")
            return

        st.info("Fetching data... Please wait.")
        data = fetch_weather_data(lat, lon, start_date, end_date, selected_vars)
        if data.empty:
            st.error("No data fetched. Check variable selection or date range.")
            return

        st.success(f"Data fetched for {len(data)} hourly records.")
        st.balloons()

        # Display a preview of the raw data (first 10 rows)
        st.markdown("---")
        st.subheader("Data Preview")
        with st.expander("Show Raw Data (first 10 rows)", expanded=False):
            st.dataframe(data.head(10))

        # ---------------------------------------------------------------
        # Update Month–Year options for Vertical Profile Explorer
        # ---------------------------------------------------------------
        month_options = sorted(data.index.strftime("%Y-%m").unique())
        if month_options:
            selected_month_year = st.sidebar.selectbox("Select Month–Year", options=month_options)

        # ---------------------------------------------------------------
        # Visualizations Section with Professional Styling
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("Visualizations")

        # Daily Time Series Chart
        if show_daily:
            fig_daily = plot_daily_timeseries(data)
            if fig_daily:
                st.plotly_chart(fig_daily, use_container_width=True)

        # Monthly Averages Chart
        if show_monthly:
            fig_monthly = plot_monthly_averages(data)
            if fig_monthly:
                st.plotly_chart(fig_monthly, use_container_width=True)

        # Soil Temperature Profile (Monthly)
        if show_soil_temp:
            fig_soil_temp = plot_soil_temperature_profile_monthly(data)
            if fig_soil_temp:
                st.plotly_chart(fig_soil_temp, use_container_width=True)

        # Soil Moisture Profile (Monthly)
        if show_soil_moist:
            fig_soil_moist = plot_soil_moisture_profile_monthly(data)
            if fig_soil_moist:
                st.plotly_chart(fig_soil_moist, use_container_width=True)

        # Vertical Profile Explorer (Interactive Chart)
        st.markdown("---")
        st.subheader("Vertical Profile Explorer")
        if selected_month_year:
            fig_profile = plot_vertical_profile(data, selected_month_year, profile_type)
            if fig_profile:
                st.plotly_chart(fig_profile, use_container_width=True)
        else:
            st.info("Vertical Profile Explorer will be available once data is fetched.")

        # ---------------------------------------------------------------
        # Display Monthly Statistics Table with Advanced Options
        # ---------------------------------------------------------------
        st.markdown("---")
        st.subheader("Monthly Statistics")
        monthly_stats = data.resample("MS").agg(["mean", "min", "max"])
        monthly_stats.columns = [f"{col}_{stat}" for col, stat in monthly_stats.columns]
        monthly_stats.reset_index(inplace=True)
        monthly_stats.rename(columns={"date": "Month"}, inplace=True)
        with st.expander("View Monthly Statistics Table", expanded=False):
            st.dataframe(monthly_stats)

        # Download CSV Option
        st.markdown("---")
        csv_data = data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="weather_data.csv",
            mime="text/csv"
        )

# ===============================================================
# RUN THE APP
# ===============================================================
if __name__ == "__main__":
    main()
