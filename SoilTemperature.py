import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import date

# -------------------------------------------------------
# PAGE CONFIG & BRANDING
# -------------------------------------------------------
st.set_page_config(
    page_title="Terracon Weather Dashboard",
    page_icon="🌎",
    layout="centered"
)
st.image(".devcontainer/Terracon-Logo 2.jpg", width=200)

# -------------------------------------------------------
# ALL POSSIBLE VARIABLES & FRIENDLY NAMES
# -------------------------------------------------------
ALL_VARS_API_TO_FRIENDLY = {
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

# Depth lookups for vertical profile
DEPTHS_TEMPERATURE = {
    "Soil Temp (0-7cm)": 3.5,
    "Soil Temp (7-28cm)": 17.5,
    "Soil Temp (28-100cm)": 64,
    "Soil Temp (100-255cm)": 177.5
}
DEPTHS_MOISTURE = {
    "Soil Moisture (0-7cm)": 3.5,
    "Soil Moisture (7-28cm)": 17.5,
    "Soil Moisture (28-100cm)": 64,
    "Soil Moisture (100-255cm)": 177.5
}

# -------------------------------------------------------
# DATA FETCHING & CACHING
# -------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_weather_data(latitude, longitude, start_dt, end_dt, selected_vars_api):
    """Fetch data from Open-Meteo API and return a DataFrame with user-friendly column names."""
    # Setup caching & retry mechanism
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # If no variables selected, return empty
    if not selected_vars_api:
        return pd.DataFrame()

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": str(start_dt),
        "end_date": str(end_dt),
        "hourly": selected_vars_api
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

    # Parse first response (archive-api returns data in a single response list)
    response = responses[0]
    hourly = response.Hourly()

    # Build time index
    date_range = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )

    data_dict = {"date": date_range}
    for i, var in enumerate(selected_vars_api):
        data_dict[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data_dict).set_index("date")

    # Rename columns to user-friendly names
    rename_map = {api_var: ALL_VARS_API_TO_FRIENDLY[api_var] for api_var in selected_vars_api}
    df.rename(columns=rename_map, inplace=True)
    return df


# -------------------------------------------------------
# HELPER: CREATE & STYLE A PLOTLY FIGURE
# -------------------------------------------------------
def create_line_chart(df, title, x_title, y_title):
    """Creates a Plotly figure from df columns, each column a line. Returns fig object."""
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col, opacity=0.8))
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
# PLOTTING FUNCTIONS (UNIFIED / REDUCED REPETITION)
# -------------------------------------------------------
def plot_timeseries(df, var_keyword, monthly=False):
    """
    Plots line charts for columns whose name contains var_keyword.
    If monthly=True, it plots monthly mean data. Otherwise, it plots raw (hourly) data.
    """
    if df.empty: 
        return None

    # Filter columns by keyword (e.g., "Temp" or "Moisture")
    cols = [c for c in df.columns if var_keyword in c]
    if not cols: 
        return None

    # If monthly, resample to monthly means
    sub_df = df[cols].copy()
    if monthly:
        sub_df = sub_df.resample("MS").mean()

    # Build chart title & y-axis based on keyword
    freq_label = "Monthly" if monthly else "Daily"
    y_label = "Temperature (°C)" if "Temp" in var_keyword else "Soil Moisture (m³/m³)"
    chart_title = f"{freq_label} {('Temperature' if 'Temp' in var_keyword else 'Soil Moisture')}"

    return create_line_chart(sub_df, chart_title, "Date", y_label)


def plot_vertical_profile(df, month_year, profile_type):
    """
    Plots a vertical profile (avg vs depth) for a given month-year.
    profile_type is either 'Temperature' or 'Moisture'.
    """
    if df.empty:
        return None

    # Filter to selected month-year
    filtered = df[df.index.strftime("%Y-%m") == month_year]
    if filtered.empty:
        st.warning("No data available for the selected month–year.")
        return None

    # Decide depths dictionary
    if profile_type == "Temperature":
        var_depth_map = DEPTHS_TEMPERATURE
    else:
        var_depth_map = DEPTHS_MOISTURE

    # Compute average for each relevant column
    avg_values = {}
    for col, depth in var_depth_map.items():
        if col in filtered.columns:
            avg_values[depth] = filtered[col].mean()

    if not avg_values:
        st.warning(f"No {profile_type.lower()} data for {month_year}.")
        return None

    # Sort by depth ascending
    depths_sorted = sorted(avg_values.keys())
    values = [avg_values[d] for d in depths_sorted]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=values, y=depths_sorted,
        mode='lines+markers',
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


def plot_profile_monthly(df, var_keyword):
    """
    Plots monthly average (line chart) of columns whose name contains var_keyword.
    Shows how the profile changes across months (but not a true vertical cross-section).
    """
    if df.empty:
        return None
    # Filter columns
    cols = [c for c in df.columns if var_keyword in c]
    if not cols:
        return None

    sub_df = df[cols].resample("MS").mean()
    y_label = "Temperature (°C)" if "Temp" in var_keyword else "Soil Moisture (m³/m³)"
    chart_title = f"Monthly {('Soil Temperature' if 'Temp' in var_keyword else 'Soil Moisture')} Profiles"
    return create_line_chart(sub_df, chart_title, "Month", y_label)


# -------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------
def main():
    # -------------
    # Sidebar Inputs
    # -------------
    st.sidebar.header("Input Parameters")
    st.sidebar.markdown("Enter location and date range. Select variables to plot.")

    # Validate lat/lon range directly via st.number_input
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=41.5998779)
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-85.3992676)

    start_date_sel = st.sidebar.date_input("Start Date", date(2010, 1, 1))
    end_date_sel = st.sidebar.date_input("End Date", date(2024, 12, 31))

    if start_date_sel > end_date_sel:
        st.error("Error: Start date cannot be after End date.")
        st.stop()

    # Sidebar variable selections
    st.sidebar.subheader("Temperature Variables")
    temp_opts = [
        "Air Temp (2m)",
        "Soil Temp (0-7cm)",
        "Soil Temp (7-28cm)",
        "Soil Temp (28-100cm)",
        "Soil Temp (100-255cm)"
    ]
    user_temp = st.sidebar.multiselect("Select Temperature Variables", temp_opts, default=temp_opts)

    st.sidebar.subheader("Soil Moisture Variables")
    moist_opts = [
        "Soil Moisture (0-7cm)",
        "Soil Moisture (7-28cm)",
        "Soil Moisture (28-100cm)",
        "Soil Moisture (100-255cm)"
    ]
    user_moist = st.sidebar.multiselect("Select Moisture Variables", moist_opts, default=moist_opts)

    # Which plots to show
    st.sidebar.markdown("#### Plots to Display:")
    show_daily_temp = st.sidebar.checkbox("Daily Temperature", value=True)
    show_daily_moist = st.sidebar.checkbox("Daily Soil Moisture", value=True)
    show_monthly_temp = st.sidebar.checkbox("Monthly Temperature", value=True)
    show_monthly_moist = st.sidebar.checkbox("Monthly Soil Moisture", value=True)
    show_temp_profile_mo = st.sidebar.checkbox("Monthly Soil Temp Profiles", value=True)
    show_moist_profile_mo = st.sidebar.checkbox("Monthly Soil Moist Profiles", value=True)

    # Convert user-friendly names back to API variable keys
    selected_vars_api = []
    for api_var, friendly_name in ALL_VARS_API_TO_FRIENDLY.items():
        if friendly_name in user_temp or friendly_name in user_moist:
            selected_vars_api.append(api_var)

    # Fetch Data button
    if st.sidebar.button("Fetch Data"):
        # Fetch data from the API
        st.info("Fetching data, please wait...")
        df_data = fetch_weather_data(lat, lon, start_date_sel, end_date_sel, selected_vars_api)

        if df_data.empty:
            st.error("No data fetched. Check date range or variables.")
            st.stop()

        # Check for large data
        if len(df_data) > 50000:
            st.warning("Large dataset retrieved—plotting may be slow.")

        st.success(f"Data fetched with {len(df_data)} records.")
        st.balloons()

        # Data Preview
        st.markdown("---")
        st.subheader("Data Preview")
        with st.expander("Show raw data (first 10 rows)", expanded=False):
            st.dataframe(df_data.head(10))
            st.write(f"Shape: {df_data.shape[0]} rows x {df_data.shape[1]} columns")

        # Visualizations
        st.markdown("---")
        st.subheader("Visualizations")
        col1, col2 = st.columns(2)

        # Daily temperature / moisture
        with col1:
            if show_daily_temp:
                fig_dt = plot_timeseries(df_data, var_keyword="Temp", monthly=False)
                if fig_dt:
                    st.plotly_chart(fig_dt, use_container_width=True)

        with col2:
            if show_daily_moist:
                fig_dm = plot_timeseries(df_data, var_keyword="Moisture", monthly=False)
                if fig_dm:
                    st.plotly_chart(fig_dm, use_container_width=True)

        # Monthly temperature / moisture
        col3, col4 = st.columns(2)
        with col3:
            if show_monthly_temp:
                fig_mt = plot_timeseries(df_data, var_keyword="Temp", monthly=True)
                if fig_mt:
                    st.plotly_chart(fig_mt, use_container_width=True)

        with col4:
            if show_monthly_moist:
                fig_mm = plot_timeseries(df_data, var_keyword="Moisture", monthly=True)
                if fig_mm:
                    st.plotly_chart(fig_mm, use_container_width=True)

        # Monthly Soil Temp/Moisture Profile
        col5, col6 = st.columns(2)
        with col5:
            if show_temp_profile_mo:
                fig_stp = plot_profile_monthly(df_data, var_keyword="Temp")
                if fig_stp:
                    st.plotly_chart(fig_stp, use_container_width=True)

        with col6:
            if show_moist_profile_mo:
                fig_smp = plot_profile_monthly(df_data, var_keyword="Moisture")
                if fig_smp:
                    st.plotly_chart(fig_smp, use_container_width=True)

        # Vertical Profile Explorer
        st.markdown("---")
        st.subheader("Vertical Profile Explorer")
        st.info("Select a month–year and profile type to view the vertical profile.")
        month_options = sorted(df_data.index.strftime("%Y-%m").unique())
        selected_month_year = st.selectbox("Select Month–Year", month_options)
        profile_type = st.radio("Profile Type", ["Temperature", "Moisture"], horizontal=True)

        if selected_month_year:
            fig_vp = plot_vertical_profile(df_data, selected_month_year, profile_type)
            if fig_vp:
                st.plotly_chart(fig_vp, use_container_width=True)

        # Monthly Statistics
        st.markdown("---")
        st.subheader("Monthly Statistics")
        monthly_stats = df_data.resample("MS").agg(["mean", "min", "max"])
        # Flatten columns (col_stat)
        monthly_stats.columns = [f"{col}_{stat}" for col, stat in monthly_stats.columns]
        monthly_stats.reset_index(inplace=True)
        monthly_stats.rename(columns={"date": "Month"}, inplace=True)

        # Rename columns for readability (5.4)
        def nice_stat_name(col_stat):
            # col_stat example: "Soil Temp (0-7cm)_mean"
            col_split = col_stat.rsplit("_", 1)
            if len(col_split) == 2:
                base, stat = col_split
                stat = stat.capitalize()
                return f"{base} ({stat})"
            return col_stat

        monthly_stats.columns = [nice_stat_name(c) for c in monthly_stats.columns]

        with st.expander("View Monthly Statistics Table", expanded=False):
            st.dataframe(monthly_stats)

        # Download Full CSV
        csv_data = df_data.to_csv().encode('utf-8')
        st.download_button(
            "Download Full Data as CSV",
            data=csv_data,
            file_name="weather_data.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>Developed by <b>Arash Hosseini</b></p>"
        "<p style='text-align: center; font-size: 0.9em;'>Data provided by Open-Meteo Archive API</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
