import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date, datetime

# ---------------------------------------------------------------
# 1) Page Configuration and Caching Setup
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Terracon Weather Dashboard",
    page_icon="🌎",
    layout="centered"
)

# Cache the data fetch to optimize performance when same parameters are used.
@st.cache_data(show_spinner=False)
def fetch_weather_data(latitude, longitude, start_date, end_date, selected_vars):
    """
    Fetches hourly weather data (including soil temperature & moisture)
    from Open-Meteo for the given latitude, longitude, date range, and
    selected variables. Returns a pandas DataFrame.
    """
    # 1) Improvement: Enhanced docstring to clarify moisture capability.
    
    # Setup caching and retry mechanism
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # 2) Improvement: Extended possible variables to include soil moisture.
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

    # Convert friendly names back to API variable names
    chosen_vars = [api_var for api_var, friendly_name in all_possible_vars.items() if friendly_name in selected_vars]
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

    # 3) Improvement: Updated chart title & y-axis label to be generic (covers soil moisture too).
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
        hovermode="x unified"
    )
    return fig

def plot_monthly_averages(df):
    """
    Plots monthly averages for each variable in the DataFrame.
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
    # 4) Improvement: Title now reflects that these can be temperature or moisture.
    fig.update_layout(
        title="Monthly Averages",
        xaxis_title="Month",
        yaxis_title="Value",
        hovermode="x unified"
    )
    return fig

def plot_monthly_distribution_box(df):
    """
    Creates a box plot of monthly average values for each variable.
    """
    if df.empty:
        return None

    monthly_avg = df.resample("MS").mean().reset_index()
    long_df = monthly_avg.melt(id_vars="date", var_name="Variable", value_name="Value")

    # 5) Improvement: More generic wording and labeling (covers moisture and temperature).
    fig = px.box(
        long_df, x="Variable", y="Value",
        title="Monthly Distribution by Depth (and Variable)",
        labels={"Variable": "Variable", "Value": "Value"}
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

def plot_soil_temp_moisture_profile_monthly(df):
    """
    Plots monthly average soil temperature and soil moisture by depth
    (two separate figures).
    """
    if df.empty:
        return None, None

    # 6) Improvement: Separate out columns for soil temperature vs. soil moisture
    soil_temp_cols = [c for c in df.columns if "soil_temperature" in c]
    soil_moist_cols = [c for c in df.columns if "soil_moisture" in c]

    # If there's no soil data, return None
    if not soil_temp_cols and not soil_moist_cols:
        return None, None

    monthly_avg = df.resample("MS").mean()

    # 7) Improvement: Provide separate figures for clarity:
    fig_temp = None
    fig_moist = None

    # Soil Temperature
    if soil_temp_cols:
        fig_temp = go.Figure()
        for col in soil_temp_cols:
            fig_temp.add_trace(go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg[col],
                mode='lines+markers',
                name=col
            ))
        fig_temp.update_layout(
            title="Monthly Average Soil Temperature by Depth",
            xaxis_title="Month",
            yaxis_title="Temperature (°C)",
            hovermode="x unified"
        )

    # Soil Moisture
    if soil_moist_cols:
        fig_moist = go.Figure()
        for col in soil_moist_cols:
            fig_moist.add_trace(go.Scatter(
                x=monthly_avg.index,
                y=monthly_avg[col],
                mode='lines+markers',
                name=col
            ))
        fig_moist.update_layout(
            title="Monthly Average Soil Moisture by Depth",
            xaxis_title="Month",
            # 8) Improvement: Soil moisture unit clarified in label
            yaxis_title="Soil Moisture (m³/m³)",
            hovermode="x unified"
        )

    return fig_temp, fig_moist

# ---------------------------------------------------------------
# 3) Streamlit App
# ---------------------------------------------------------------
def main():
    # 9) Improvement: Simplified developer credit formatting & text.
    st.markdown(
        "<h3 style='text-align: center;'>Developed by <b>Arash Hosseini</b></h3>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.title("Weather Dashboard")

    # Sidebar About Section
    with st.sidebar.expander("ℹ️ About this App", expanded=False):
        st.write(
            """
            This interactive weather dashboard fetches historical Open-Meteo data for a 
            specified latitude and longitude over a custom date range. 
            You can select variables (including soil temperature & moisture) and view 
            both daily and monthly statistics and visualizations.
            """
        )

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    lat = st.sidebar.number_input(
        "Latitude",
        value=41.5998779,
        help="Enter the latitude for your location. Example: 41.5998779"
    )
    lon = st.sidebar.number_input(
        "Longitude",
        value=-85.3992676,
        help="Enter the longitude for your location. Example: -85.3992676"
    )
    
    # 10) Improvement: More practical default date range (slightly narrower).
    start_date = st.sidebar.date_input(
        "Start Date",
        date(2010, 1, 1),
        help="Select the earliest date in your desired range."
    )
    end_date = st.sidebar.date_input(
        "End Date",
        date(2024, 12, 31),
        help="Select the latest date in your desired range."
    )

    # Variable selection
    possible_vars = [
        "Air Temp (2m)",
        "Soil Temp (0-7cm)",
        "Soil Temp (7-28cm)",
        "Soil Temp (28-100cm)",
        "Soil Temp (100-255cm)",
        "Soil Moisture (0-7cm)",
        "Soil Moisture (7-28cm)",
        "Soil Moisture (28-100cm)",
        "Soil Moisture (100-255cm)"
    ]
    selected_vars = st.sidebar.multiselect(
        "Select Variables",
        options=possible_vars,
        default=possible_vars
    )

    # Let users choose which plots to display
    st.sidebar.markdown("#### Choose Plots to Display:")
    show_daily = st.sidebar.checkbox("Daily Time Series", value=True)
    show_monthly_avg = st.sidebar.checkbox("Monthly Averages", value=True)
    show_box = st.sidebar.checkbox("Monthly Box Plot", value=True)

    # REMOVED correlation heatmap entirely (Issue removed #1).

    # Fetch Data button
    if st.sidebar.button("Fetch Data"):
        # 2) Issue removed: Proper date range validation that won't break if same day is chosen.
        if start_date > end_date:
            st.error("Error: Start date must be before or equal to End date.")
            return

        # Info message while fetching data
        st.info("Fetching data... Please wait.")
        data = fetch_weather_data(lat, lon, start_date, end_date, selected_vars)
        if data.empty:
            st.error("No data fetched. Check variable selection or date range.")
            return

        st.success(f"Data fetched for {len(data)} hourly records.")
        st.balloons()

        st.markdown("---")
        st.subheader("Data Preview")
        # 3) Issue removed: Show entire dataframe might be too big; just show 10 rows.
        with st.expander("Show Raw Data (first 10 rows)", expanded=False):
            st.dataframe(data.head(10))

        # --- Plots ---
        st.markdown("---")
        st.subheader("Visualizations")

        # Daily Time Series
        if show_daily:
            fig_daily = plot_daily_timeseries(data)
            if fig_daily:
                st.plotly_chart(fig_daily)

        # Monthly Averages
        if show_monthly_avg:
            fig_monthly = plot_monthly_averages(data)
            if fig_monthly:
                st.plotly_chart(fig_monthly)

        # Box Plot
        if show_box:
            fig_box = plot_monthly_distribution_box(data)
            if fig_box:
                st.plotly_chart(fig_box)

        # Soil Temperature & Moisture Profiles by Depth (monthly avg)
        soil_temp_fig, soil_moist_fig = plot_soil_temp_moisture_profile_monthly(data)
        if soil_temp_fig or soil_moist_fig:
            st.markdown("---")
            st.subheader("Soil Profiles (Monthly Averages)")
            if soil_temp_fig:
                st.plotly_chart(soil_temp_fig)
            if soil_moist_fig:
                st.plotly_chart(soil_moist_fig)

        # Monthly Stats Table
        st.markdown("---")
        st.subheader("Monthly Statistics")
        monthly_stats_df = create_monthly_stats_table(data)
        if monthly_stats_df.empty:
            st.info("No data to calculate monthly stats.")
        else:
            with st.expander("View Monthly Statistics Table", expanded=False):
                st.dataframe(monthly_stats_df)

        # Download CSV
        st.markdown("---")
        csv_data = data.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="weather_data.csv",
            mime="text/csv"
        )

# Run the app
if __name__ == "__main__":
    main()
