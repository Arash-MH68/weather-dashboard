import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date

# ===============================================================
# PAGE CONFIGURATION & BRANDING
# ===============================================================
st.set_page_config(
    page_title="Terracon Weather Dashboard",
    page_icon="🌎",
    layout="centered"
)

# Terracon Logo for branding (improvement: visible brand identity)
st.image("https://www.terracon.com/wp-content/uploads/2017/08/Terracon-logo.png", width=200)

# ===============================================================
# DATA FETCHING & CACHING FUNCTION
# ===============================================================
@st.cache_data(show_spinner=False)
def fetch_weather_data(latitude, longitude, start_date, end_date, selected_vars):
    """
    Fetch hourly weather data from Open-Meteo including both soil temperature and soil moisture,
    based on the selected variables.
    """
    # Setup caching and retry mechanism for reliability.
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # All possible variables with their friendly names.
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

    # Convert friendly names from user selection into API variable names.
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

    # Create a time index from API parameters.
    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    # Build the data dictionary.
    data_dict = {"date": date_range}
    for i, var in enumerate(chosen_vars):
        data_dict[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data_dict)
    df.set_index("date", inplace=True)
    return df

# ===============================================================
# PLOTTING FUNCTIONS
# ===============================================================
def plot_daily_temperature_timeseries(df):
    """Plot daily time series for temperature variables only."""
    if df.empty:
        return None
    # Filter for temperature: include Air Temp and soil_temperature variables.
    temp_cols = [col for col in df.columns if col == "temperature_2m" or "soil_temperature" in col]
    if not temp_cols:
        return None
    fig = go.Figure()
    for col in temp_cols:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines',
            name=col,
            opacity=0.8
        ))
    fig.update_layout(
        title="Daily Temperature Time Series",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_daily_moisture_timeseries(df):
    """Plot daily time series for soil moisture variables only."""
    if df.empty:
        return None
    moist_cols = [col for col in df.columns if "soil_moisture" in col]
    if not moist_cols:
        return None
    fig = go.Figure()
    for col in moist_cols:
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col],
            mode='lines',
            name=col,
            opacity=0.8
        ))
    fig.update_layout(
        title="Daily Soil Moisture Time Series",
        xaxis_title="Date",
        yaxis_title="Soil Moisture (m³/m³)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_monthly_temperature_averages(df):
    """Plot monthly average temperature for temperature variables."""
    if df.empty:
        return None
    temp_cols = [col for col in df.columns if col == "temperature_2m" or "soil_temperature" in col]
    if not temp_cols:
        return None
    monthly_avg = df[temp_cols].resample("MS").mean()
    fig = go.Figure()
    for col in monthly_avg.columns:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=col
        ))
    fig.update_layout(
        title="Monthly Average Temperature",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_monthly_moisture_averages(df):
    """Plot monthly average soil moisture for moisture variables."""
    if df.empty:
        return None
    moist_cols = [col for col in df.columns if "soil_moisture" in col]
    if not moist_cols:
        return None
    monthly_avg = df[moist_cols].resample("MS").mean()
    fig = go.Figure()
    for col in monthly_avg.columns:
        fig.add_trace(go.Scatter(
            x=monthly_avg.index, y=monthly_avg[col],
            mode='lines+markers',
            name=col
        ))
    fig.update_layout(
        title="Monthly Average Soil Moisture",
        xaxis_title="Month",
        yaxis_title="Soil Moisture (m³/m³)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_soil_temperature_profile_monthly(df):
    """Plot monthly average soil temperature profiles (by depth)."""
    if df.empty:
        return None
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
        title="Monthly Soil Temperature Profiles",
        xaxis_title="Month",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_soil_moisture_profile_monthly(df):
    """Plot monthly average soil moisture profiles (by depth)."""
    if df.empty:
        return None
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
        title="Monthly Soil Moisture Profiles",
        xaxis_title="Month",
        yaxis_title="Soil Moisture (m³/m³)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

def plot_vertical_profile(df, month_year, profile_type):
    """
    Plot a vertical profile (average vs. depth) for a selected month-year.
    Depth mapping (in cm) is defined for each soil variable.
    """
    if df.empty:
        return None

    # Filter data for the selected month–year.
    filtered = df[df.index.strftime("%Y-%m") == month_year]
    if filtered.empty:
        st.warning("No data available for the selected month–year.")
        return None

    # Define depth mappings for temperature and moisture (average depth in cm).
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

    vars_depth = depths_temperature if profile_type == "Temperature" else depths_moisture

    avg_values = {}
    for var, depth in vars_depth.items():
        if var in filtered.columns:
            avg_values[depth] = filtered[var].mean()

    if not avg_values:
        st.warning(f"No {profile_type.lower()} data available for the selected month–year.")
        return None

    # Sort by depth (ascending) so that depth increases downward.
    depths_sorted = sorted(avg_values.keys())
    values = [avg_values[d] for d in depths_sorted]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=depths_sorted,
        mode='lines+markers',
        line=dict(width=2),
        marker=dict(size=10),
        name=f"{profile_type} Profile"
    ))
    fig.update_layout(
        title=f"{profile_type} Vertical Profile for {month_year}",
        xaxis_title=f"{profile_type} Value",
        yaxis_title="Depth (cm)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        hovermode="y"
    )
    return fig

# ===============================================================
# STREAMLIT APP: INPUT & OUTPUT SECTIONS
# ===============================================================
def main():
    # -----------------------------
    # Sidebar Input Section (Improved)
    # -----------------------------
    st.sidebar.header("Input Parameters")
    st.sidebar.markdown("Enter location details and select variables for data extraction.")
    lat = st.sidebar.number_input("Latitude", value=41.5998779, help="Enter latitude (e.g., 41.5998779)")
    lon = st.sidebar.number_input("Longitude", value=-85.3992676, help="Enter longitude (e.g., -85.3992676)")
    start_date = st.sidebar.date_input("Start Date", date(2010, 1, 1), help="Select start date")
    end_date = st.sidebar.date_input("End Date", date(2024, 12, 31), help="Select end date")

    # Separate variable selections for Temperature and Soil Moisture.
    st.sidebar.subheader("Temperature Variables")
    temp_options = ["Air Temp (2m)", "Soil Temp (0-7cm)", "Soil Temp (7-28cm)",
                    "Soil Temp (28-100cm)", "Soil Temp (100-255cm)"]
    selected_temp = st.sidebar.multiselect("Select Temperature Variables", options=temp_options, default=temp_options)
    
    st.sidebar.subheader("Soil Moisture Variables")
    moist_options = ["Soil Moisture (0-7cm)", "Soil Moisture (7-28cm)",
                     "Soil Moisture (28-100cm)", "Soil Moisture (100-255cm)"]
    selected_moist = st.sidebar.multiselect("Select Moisture Variables", options=moist_options, default=moist_options)
    
    # Combine selections for the API query.
    selected_vars = selected_temp + selected_moist

    # Plot display options.
    st.sidebar.markdown("#### Select Plots to Display:")
    show_daily_temp = st.sidebar.checkbox("Daily Temperature", value=True)
    show_daily_moist = st.sidebar.checkbox("Daily Soil Moisture", value=True)
    show_monthly_temp = st.sidebar.checkbox("Monthly Temperature", value=True)
    show_monthly_moist = st.sidebar.checkbox("Monthly Soil Moisture", value=True)
    show_soil_temp_profile = st.sidebar.checkbox("Soil Temperature Profile (Monthly)", value=True)
    show_soil_moist_profile = st.sidebar.checkbox("Soil Moisture Profile (Monthly)", value=True)

    # Button to fetch data.
    if st.sidebar.button("Fetch Data"):
        if start_date > end_date:
            st.error("Error: Start date must be before or equal to End date.")
            return

        st.info("Fetching data... Please wait.")
        data = fetch_weather_data(lat, lon, start_date, end_date, selected_vars)
        if data.empty:
            st.error("No data fetched. Check your variable selection or date range.")
            return

        st.success(f"Data fetched with {len(data)} hourly records.")
        st.balloons()

        # -----------------------------
        # Output Section: Data Preview & Graphs
        # -----------------------------
        st.markdown("---")
        st.subheader("Data Preview")
        with st.expander("Show Raw Data (first 10 rows)", expanded=False):
            st.dataframe(data.head(10))
        
        # Organize graphs in two columns for a professional layout.
        st.markdown("---")
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        # Daily Time Series: Temperature and Moisture (separated)
        with col1:
            if show_daily_temp:
                fig_dt = plot_daily_temperature_timeseries(data)
                if fig_dt:
                    st.plotly_chart(fig_dt, use_container_width=True)
        with col2:
            if show_daily_moist:
                fig_dm = plot_daily_moisture_timeseries(data)
                if fig_dm:
                    st.plotly_chart(fig_dm, use_container_width=True)
        
        # Monthly Averages: Temperature and Moisture
        col3, col4 = st.columns(2)
        with col3:
            if show_monthly_temp:
                fig_mt = plot_monthly_temperature_averages(data)
                if fig_mt:
                    st.plotly_chart(fig_mt, use_container_width=True)
        with col4:
            if show_monthly_moist:
                fig_mm = plot_monthly_moisture_averages(data)
                if fig_mm:
                    st.plotly_chart(fig_mm, use_container_width=True)

        # Soil Profiles: Temperature and Moisture (Monthly)
        col5, col6 = st.columns(2)
        with col5:
            if show_soil_temp_profile:
                fig_stp = plot_soil_temperature_profile_monthly(data)
                if fig_stp:
                    st.plotly_chart(fig_stp, use_container_width=True)
        with col6:
            if show_soil_moist_profile:
                fig_smp = plot_soil_moisture_profile_monthly(data)
                if fig_smp:
                    st.plotly_chart(fig_smp, use_container_width=True)
        
        # -----------------------------
        # Vertical Profile Explorer (Output Side Only)
        # -----------------------------
        st.markdown("---")
        st.subheader("Vertical Profile Explorer")
        st.info("Select a month–year and profile type below to view the vertical profile. This update does not require re-fetching data.")
        # Get unique month–year options from data.
        month_options = sorted(data.index.strftime("%Y-%m").unique())
        selected_month_year = st.selectbox("Select Month–Year", options=month_options)
        profile_type = st.radio("Profile Type", options=["Temperature", "Moisture"], horizontal=True)
        if selected_month_year:
            fig_vp = plot_vertical_profile(data, selected_month_year, profile_type)
            if fig_vp:
                st.plotly_chart(fig_vp, use_container_width=True)
        
        # -----------------------------
        # Monthly Statistics Table (with download option)
        # -----------------------------
        st.markdown("---")
        st.subheader("Monthly Statistics")
        monthly_stats = data.resample("MS").agg(["mean", "min", "max"])
        monthly_stats.columns = [f"{col}_{stat}" for col, stat in monthly_stats.columns]
        monthly_stats.reset_index(inplace=True)
        monthly_stats.rename(columns={"date": "Month"}, inplace=True)
        with st.expander("View Monthly Statistics Table", expanded=False):
            st.dataframe(monthly_stats)
        
        csv_data = data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Full Data as CSV",
            data=csv_data,
            file_name="weather_data.csv",
            mime="text/csv"
        )
    
    # Footer with developer credit.
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Developed by <b>Arash Hosseini</b></p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 0.9em;'>Data provided by Open-Meteo Archive API</p>", unsafe_allow_html=True)

# ===============================================================
# RUN THE APP
# ===============================================================
if __name__ == "__main__":
    main()
