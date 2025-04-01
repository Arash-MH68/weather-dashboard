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
st.set_page_config(page_title="Terracon Weather Dashboard", page_icon="🌎", layout="centered")
st.image(".devcontainer/Terracon-Logo 2.jpg", width=200)

# -------------------------------------------------------
# CONSTANTS & LOOKUPS
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
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

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

    response = responses[0]
    hourly = response.Hourly()

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

    # Rename columns
    rename_map = {api_var: ALL_VARS_API_TO_FRIENDLY[api_var] for api_var in selected_vars_api}
    df.rename(columns=rename_map, inplace=True)
    return df


# -------------------------------------------------------
# CHART BUILDING HELPERS
# -------------------------------------------------------
def create_plotly_line_chart(
    df,
    title="Time Series",
    x_title="Date",
    y_title="Value",
    fig_width=900,
    fig_height=500,
    line_width=2,
    marker_size=4,
    show_legend=True,
    yaxis_range=None
):
    """Generic line chart builder for multiple columns of df."""
    if df.empty:
        return None

    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines+markers",
                name=col,
                line=dict(width=line_width),
                marker=dict(size=marker_size),
                connectgaps=True
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="x unified",
        template="plotly_white",
        width=fig_width,
        height=fig_height,
        showlegend=show_legend,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    if yaxis_range and len(yaxis_range) == 2:
        fig.update_yaxes(range=yaxis_range)

    return fig


def apply_rolling_average(df, window=24):
    """Apply a rolling average with the specified window (in hours, by default)."""
    return df.rolling(window=window, min_periods=1).mean()


def resample_data(df, freq="D"):
    """
    Resample data to specified freq.
    freq options: 'H' (hourly), 'D' (daily), 'W' (weekly), 'MS' (monthly).
    """
    return df.resample(freq).mean()


# -------------------------------------------------------
# PLOTTING FUNCTIONS
# -------------------------------------------------------
def plot_timeseries(
    df,
    var_keyword,
    frequency,
    rolling=False,
    rolling_window=24,
    fig_width=900,
    fig_height=500,
    line_width=2,
    marker_size=4,
    show_legend=True,
    yaxis_range=None
):
    """
    Generate a time series for columns containing var_keyword,
    allowing user to pick frequency (H, D, W, MS), optionally apply rolling average.
    """
    if df.empty:
        return None

    cols = [c for c in df.columns if var_keyword in c]
    if not cols:
        return None

    sub_df = df[cols].copy()
    sub_df = resample_data(sub_df, freq=frequency) if frequency != "H" else sub_df

    if rolling:
        sub_df = apply_rolling_average(sub_df, window=rolling_window)

    y_label = "Temperature (°C)" if "Temp" in var_keyword else "Soil Moisture (m³/m³)"
    chart_title = f"{'Temperature' if 'Temp' in var_keyword else 'Soil Moisture'} ({frequency} freq)"

    return create_plotly_line_chart(
        sub_df, 
        title=chart_title,
        y_title=y_label,
        fig_width=fig_width,
        fig_height=fig_height,
        line_width=line_width,
        marker_size=marker_size,
        show_legend=show_legend,
        yaxis_range=yaxis_range
    )


def plot_vertical_profile(
    df,
    month_year,
    profile_type,
    fig_width=900,
    fig_height=500,
    line_width=2,
    marker_size=4,
    show_legend=True
):
    """Plots a vertical average vs. depth for the chosen month-year."""
    if df.empty:
        return None

    filtered = df[df.index.strftime("%Y-%m") == month_year]
    if filtered.empty:
        st.warning("No data for that month–year.")
        return None

    var_depth_map = DEPTHS_TEMPERATURE if profile_type == "Temperature" else DEPTHS_MOISTURE
    avg_values = {}

    for col, depth in var_depth_map.items():
        if col in filtered.columns:
            avg_values[depth] = filtered[col].mean()

    if not avg_values:
        st.warning(f"No {profile_type.lower()} data in that month–year.")
        return None

    depths_sorted = sorted(avg_values.keys())
    vals_sorted = [avg_values[d] for d in depths_sorted]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vals_sorted,
            y=depths_sorted,
            mode="lines+markers",
            name=f"{profile_type} Profile",
            line=dict(width=line_width),
            marker=dict(size=marker_size)
        )
    )

    fig.update_layout(
        title=f"{profile_type} Vertical Profile for {month_year}",
        xaxis_title=f"{profile_type} Value",
        yaxis_title="Depth (cm)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        hovermode="y",
        width=fig_width,
        height=fig_height,
        showlegend=show_legend,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------
def main():
    st.sidebar.header("Input Parameters")
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=41.5998779)
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-85.3992676)
    start_date_sel = st.sidebar.date_input("Start Date", date(2010, 1, 1))
    end_date_sel = st.sidebar.date_input("End Date", date(2024, 12, 31))
    if start_date_sel > end_date_sel:
        st.error("Start date cannot be after end date.")
        st.stop()

    st.sidebar.subheader("Temperature Vars")
    temp_opts = [
        "Air Temp (2m)",
        "Soil Temp (0-7cm)",
        "Soil Temp (7-28cm)",
        "Soil Temp (28-100cm)",
        "Soil Temp (100-255cm)"
    ]
    user_temp = st.sidebar.multiselect("Select Temperature", temp_opts, default=temp_opts)

    st.sidebar.subheader("Soil Moisture Vars")
    moist_opts = [
        "Soil Moisture (0-7cm)",
        "Soil Moisture (7-28cm)",
        "Soil Moisture (28-100cm)",
        "Soil Moisture (100-255cm)"
    ]
    user_moist = st.sidebar.multiselect("Select Moisture", moist_opts, default=moist_opts)

    # Convert to API var keys
    selected_vars_api = []
    for api_var, friendly in ALL_VARS_API_TO_FRIENDLY.items():
        if friendly in user_temp or friendly in user_moist:
            selected_vars_api.append(api_var)

    st.sidebar.subheader("Data Fetch / Refresh")
    if st.sidebar.button("Fetch / Refresh Data"):
        st.info("Querying API...")
        df_data = fetch_weather_data(lat, lon, start_date_sel, end_date_sel, selected_vars_api)
        if df_data.empty:
            st.error("No data returned.")
            st.stop()
        st.session_state["df_data"] = df_data
        st.success(f"Data fetched: {len(df_data)} records.")
        st.balloons()

    # Ensure df_data is in session_state
    if "df_data" not in st.session_state or st.session_state["df_data"].empty:
        st.warning("No data available. Please fetch data first.")
        st.stop()

    df_data = st.session_state["df_data"]
    if len(df_data) > 50000:
        st.warning("Large dataset; you may experience some slowdown with many charts.")

    # -----------------------------------------------------
    # Sidebar: Visualization Controls
    # -----------------------------------------------------
    st.sidebar.subheader("Visualization Controls")
    freq_map = {"Hourly": "H", "Daily": "D", "Weekly": "W", "Monthly": "MS"}
    chosen_freq_temp = st.sidebar.selectbox("Temp Time Aggregation", list(freq_map.keys()), index=1)  # Default: Daily
    chosen_freq_moist = st.sidebar.selectbox("Moist Time Aggregation", list(freq_map.keys()), index=1)

    rolling_avg = st.sidebar.checkbox("Apply Rolling Average?", value=False)
    rolling_window = st.sidebar.number_input("Rolling Window (hours or aggregated steps)", min_value=1, max_value=168, value=24)

    fig_width = st.sidebar.slider("Figure Width (px)", 600, 1600, 1000)
    fig_height = st.sidebar.slider("Figure Height (px)", 400, 900, 550)
    line_width = st.sidebar.slider("Line Thickness", 1, 6, 2)
    marker_size = st.sidebar.slider("Marker Size", 2, 10, 4)
    show_legend = st.sidebar.checkbox("Show Legend?", value=True)

    manual_y_range = st.sidebar.checkbox("Manual Y-axis Range?", value=False)
    y_min = st.sidebar.number_input("Y-axis Min", value=0.0) if manual_y_range else None
    y_max = st.sidebar.number_input("Y-axis Max", value=40.0) if manual_y_range else None
    y_range = [y_min, y_max] if manual_y_range else None

    # -----------------------------------------------------
    # Data Preview
    # -----------------------------------------------------
    st.subheader("Data Preview")
    with st.expander("Show raw data (first 10 rows)", expanded=False):
        st.dataframe(df_data.head(10))
        st.write(f"Shape: {df_data.shape}")

    # -----------------------------------------------------
    # Visualizations
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Visualizations")

    # Time Series: Temperature
    if any("Temp" in c for c in df_data.columns):
        st.write("#### Temperature Time Series")
        fig_temp = plot_timeseries(
            df_data,
            var_keyword="Temp",
            frequency=freq_map[chosen_freq_temp],
            rolling=rolling_avg,
            rolling_window=rolling_window,
            fig_width=fig_width,
            fig_height=fig_height,
            line_width=line_width,
            marker_size=marker_size,
            show_legend=show_legend,
            yaxis_range=y_range
        )
        if fig_temp:
            st.plotly_chart(fig_temp, use_container_width=False)

    # Time Series: Moisture
    if any("Moisture" in c for c in df_data.columns):
        st.write("#### Soil Moisture Time Series")
        fig_moist = plot_timeseries(
            df_data,
            var_keyword="Moisture",
            frequency=freq_map[chosen_freq_moist],
            rolling=rolling_avg,
            rolling_window=rolling_window,
            fig_width=fig_width,
            fig_height=fig_height,
            line_width=line_width,
            marker_size=marker_size,
            show_legend=show_legend,
            yaxis_range=y_range
        )
        if fig_moist:
            st.plotly_chart(fig_moist, use_container_width=False)

    # -----------------------------------------------------
    # Vertical Profile Explorer
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Vertical Profile Explorer")
    month_options = sorted(df_data.index.strftime("%Y-%m").unique())
    if month_options:
        chosen_month = st.selectbox("Choose Month–Year", month_options)
        prof_type = st.radio("Profile Type", ["Temperature", "Moisture"], horizontal=True)
        fig_prof = plot_vertical_profile(
            df_data, chosen_month, prof_type,
            fig_width=fig_width, fig_height=fig_height,
            line_width=line_width, marker_size=marker_size,
            show_legend=show_legend
        )
        if fig_prof:
            st.plotly_chart(fig_prof, use_container_width=False)

    # -----------------------------------------------------
    # Monthly Statistics
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Monthly Statistics")
    monthly_stats = df_data.resample("MS").agg(["mean", "min", "max"])
    monthly_stats.columns = [f"{col}_{stat}" for col, stat in monthly_stats.columns]
    monthly_stats.reset_index(inplace=True)
    monthly_stats.rename(columns={"date": "Month"}, inplace=True)

    # Make table columns friendlier
    def nice_stat_name(col_stat):
        parts = col_stat.rsplit("_", 1)
        if len(parts) == 2:
            base, stat = parts
            return f"{base} ({stat.capitalize()})"
        return col_stat

    monthly_stats.columns = [nice_stat_name(c) for c in monthly_stats.columns]

    with st.expander("View Monthly Statistics Table", expanded=False):
        st.dataframe(monthly_stats)

    # Download Full CSV
    csv_data = df_data.to_csv().encode("utf-8")
    st.download_button("Download Full Data as CSV", data=csv_data, file_name="weather_data.csv", mime="text/csv")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>Developed by <b>Arash Hosseini</b></p>"
        "<p style='text-align: center; font-size: 0.9em;'>Data provided by Open-Meteo Archive API</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
