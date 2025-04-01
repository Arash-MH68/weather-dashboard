import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

    # Rename columns to user-friendly
    rename_map = {api_var: ALL_VARS_API_TO_FRIENDLY[api_var] for api_var in selected_vars_api}
    df.rename(columns=rename_map, inplace=True)
    return df


# -------------------------------------------------------
# UTILITY: FREEZE ANALYSIS
# -------------------------------------------------------
def compute_freeze_stats(sub_df: pd.DataFrame):
    """
    For each temperature column in sub_df, compute:
      - total hours (or days) below freezing (0°C)
      - earliest freeze time and latest freeze time
      - min / max temperature
    Return a dict of dicts keyed by column name.
    """
    freeze_info = {}
    for col in sub_df.columns:
        if "Temp" not in col:
            continue
        # For each row below 0
        below_freeze = sub_df[sub_df[col] < 0]
        freeze_hours = len(below_freeze)  # raw count of timesteps, can interpret as hours/days
        if not below_freeze.empty:
            earliest_freeze = below_freeze.index[0]
            latest_freeze = below_freeze.index[-1]
        else:
            earliest_freeze = None
            latest_freeze = None

        min_val = sub_df[col].min()
        max_val = sub_df[col].max()

        freeze_info[col] = {
            "BelowFreezingCount": freeze_hours,
            "EarliestFreeze": earliest_freeze,
            "LatestFreeze": latest_freeze,
            "MinTemp": min_val,
            "MaxTemp": max_val
        }
    return freeze_info


def display_freeze_stats(freeze_stats: dict, frequency_label: str):
    """
    Nicely display freeze stats for each column in the right column.
    Frequency label is 'Hourly' or 'Daily' to help interpret 'BelowFreezingCount'.
    """
    if not freeze_stats:
        st.write("No temperature columns selected or no below-freezing data.")
        return

    for col, info in freeze_stats.items():
        st.markdown(f"#### {col}")
        bf_count = info["BelowFreezingCount"]
        earliest = info["EarliestFreeze"]
        latest = info["LatestFreeze"]
        if earliest and latest:
            st.write(
                f"- **Time below freezing:** {bf_count} {frequency_label.lower()} data points\n"
                f"- **Earliest freeze:** {earliest}\n"
                f"- **Latest freeze:** {latest}"
            )
        else:
            st.write(f"- **Time below freezing:** {bf_count} {frequency_label.lower()} data points\n")
        st.write(
            f"- **Min Temp:** {info['MinTemp']:.2f}°C\n"
            f"- **Max Temp:** {info['MaxTemp']:.2f}°C"
        )
        st.markdown("---")


# -------------------------------------------------------
# CHART BUILDING
# -------------------------------------------------------
def create_plotly_line_chart(
    df: pd.DataFrame,
    title: str,
    y_title: str,
    is_temperature: bool = False,
    width: int = 1200,
    height: int = 600
):
    """Plotly line chart with black axis lines/grid and a horizontal 0°C line if temperature."""
    if df.empty:
        return None

    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines+markers",
                name=col
                # default color cycle for lines
            )
        )

    # If temperature, add freezing line at 0°C
    if is_temperature:
        fig.add_hline(
            y=0,
            line=dict(width=2, color="black", dash="dash"),
            annotation_text="Freezing (0°C)",
            annotation_position="top left"
        )

    # Style
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        template="plotly_white",
        width=width,
        height=height,
        showlegend=True,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    # Force black axis lines and grid lines
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="black")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="black")
    return fig


def plot_timeseries(df: pd.DataFrame, columns_to_plot, freq: str = "H"):
    """
    Plots either hourly or daily data for the given columns.
    freq can be 'H' or 'D'.
    """
    if not columns_to_plot:
        return None

    sub_df = df[columns_to_plot].copy()
    if freq == "D":
        sub_df = sub_df.resample("D").mean()

    # Determine if we have temperature in columns
    is_temp = any("Temp" in c for c in columns_to_plot)
    y_label = "Temperature (°C)" if is_temp else "Soil Moisture (m³/m³)"
    freq_label = "Daily" if freq == "D" else "Hourly"
    chart_title = f"{freq_label} {'Temperature' if is_temp else 'Soil Moisture'}"

    fig = create_plotly_line_chart(sub_df, title=chart_title, y_title=y_label, is_temperature=is_temp)
    return fig, sub_df


def plot_vertical_profile(df: pd.DataFrame, month_year: str, profile_type: str):
    """Plots a vertical average vs. depth for chosen month-year, with a dashed line at 0°C if temperature."""
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
            name=f"{profile_type} Profile"
        )
    )

    # If temperature, add freezing line. We'll do it horizontally if possible,
    # but the x-axis is temperature so we must check if 0 is in range:
    if profile_type == "Temperature":
        fig.add_vline(
            x=0,
            line=dict(width=2, color="black", dash="dash"),
            annotation_text="Freezing (0°C)",
            annotation_position="bottom right"
        )

    fig.update_layout(
        title=f"{profile_type} Vertical Profile for {month_year}",
        xaxis_title=f"{profile_type} Value (°C)" if profile_type == "Temperature" else f"{profile_type} Value (m³/m³)",
        yaxis_title="Depth (cm)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        hovermode="y",
        width=1200,
        height=600,
        showlegend=True,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    # Force black axis lines and grid lines
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="black")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="black")

    return fig


# -------------------------------------------------------
# MAIN APP
# -------------------------------------------------------
def main():
    # -----------------------------
    # Sidebar Inputs
    # -----------------------------
    st.sidebar.header("Input Parameters")
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=41.5998779)
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-85.3992676)
    start_date_sel = st.sidebar.date_input("Start Date", date(2010, 1, 1))
    end_date_sel = st.sidebar.date_input("End Date", date(2024, 12, 31))
    if start_date_sel > end_date_sel:
        st.error("Start date cannot be after end date.")
        st.stop()

    # Temperature & Moisture selections
    st.sidebar.subheader("Temperature Variables")
    temp_opts = [
        "Air Temp (2m)",
        "Soil Temp (0-7cm)",
        "Soil Temp (7-28cm)",
        "Soil Temp (28-100cm)",
        "Soil Temp (100-255cm)"
    ]
    user_temp = st.sidebar.multiselect("Select Temperature Vars", temp_opts, default=temp_opts)

    st.sidebar.subheader("Soil Moisture Variables")
    moist_opts = [
        "Soil Moisture (0-7cm)",
        "Soil Moisture (7-28cm)",
        "Soil Moisture (28-100cm)",
        "Soil Moisture (100-255cm)"
    ]
    user_moist = st.sidebar.multiselect("Select Moisture Vars", moist_opts, default=moist_opts)

    # Convert user-friendly names back to API
    selected_vars_api = []
    for api_var, friendly in ALL_VARS_API_TO_FRIENDLY.items():
        if friendly in user_temp or friendly in user_moist:
            selected_vars_api.append(api_var)

    # -----------------------------
    # Fetch Data
    # -----------------------------
    st.sidebar.subheader("Data Fetch / Refresh")
    if st.sidebar.button("Fetch / Refresh Data"):
        st.info("Querying API... please wait.")
        df_data = fetch_weather_data(lat, lon, start_date_sel, end_date_sel, selected_vars_api)
        if df_data.empty:
            st.error("No data returned.")
            st.stop()
        st.session_state["df_data"] = df_data
        st.success(f"Data fetched: {len(df_data)} records.")
        st.balloons()

    # If no data in session, stop
    if "df_data" not in st.session_state or st.session_state["df_data"].empty:
        st.warning("No data available. Please fetch data first.")
        st.stop()

    df_data = st.session_state["df_data"]
    if len(df_data) > 50000:
        st.warning("Large dataset; you may experience some slowdown.")

    # -----------------------------------------------------
    # Let user pick which columns to plot more precisely
    # -----------------------------------------------------
    st.sidebar.subheader("Select Which Columns to Display")
    # Filter the columns present in df_data
    available_temp_cols = [c for c in df_data.columns if "Temp" in c]
    available_moist_cols = [c for c in df_data.columns if "Moisture" in c]

    chosen_temp_cols = st.sidebar.multiselect(
        "Temperature columns to show on chart",
        available_temp_cols,
        default=available_temp_cols
    )
    chosen_moist_cols = st.sidebar.multiselect(
        "Moisture columns to show on chart",
        available_moist_cols,
        default=available_moist_cols
    )

    # Let user pick hourly or daily
    st.sidebar.subheader("Aggregation Frequency")
    freq_temp = st.sidebar.radio("Temperature freq:", ["Hourly", "Daily"], horizontal=True)
    freq_moist = st.sidebar.radio("Moisture freq:", ["Hourly", "Daily"], horizontal=True)

    # -----------------------------------------------------
    # Data Preview
    # -----------------------------------------------------
    st.subheader("Data Preview")
    with st.expander("Show raw data (first 10 rows)", expanded=False):
        st.dataframe(df_data.head(10))
        st.write(f"Shape: {df_data.shape}")

    # -----------------------------------------------------
    # Temperature Time Series
    # -----------------------------------------------------
    if chosen_temp_cols:
        st.markdown("---")
        st.write("### Temperature Time Series")
        col1, col2 = st.columns([2,1])  # chart in col1, stats in col2

        with col1:
            freq_code = "D" if freq_temp == "Daily" else "H"
            fig_temp, df_temp_plotted = plot_timeseries(df_data, chosen_temp_cols, freq_code)
            if fig_temp:
                st.plotly_chart(fig_temp, use_container_width=False)

        with col2:
            # Freeze stats
            freeze_stats = compute_freeze_stats(df_temp_plotted)
            st.write("#### Freeze Analysis")
            st.caption("Approximate freeze times based on data in the plot.")
            display_freeze_stats(freeze_stats, frequency_label=freq_temp)

    # -----------------------------------------------------
    # Moisture Time Series
    # -----------------------------------------------------
    if chosen_moist_cols:
        st.markdown("---")
        st.write("### Soil Moisture Time Series")
        col3, col4 = st.columns([2,1])

        with col3:
            freq_code = "D" if freq_moist == "Daily" else "H"
            fig_moist, df_moist_plotted = plot_timeseries(df_data, chosen_moist_cols, freq_code)
            if fig_moist:
                st.plotly_chart(fig_moist, use_container_width=False)

        with col4:
            # Moisture doesn't have below-freezing logic, but we could add some note
            st.write("#### Moisture Info")
            st.caption("Below-freezing line is irrelevant for moisture, no freeze analysis shown.")
            st.write(f"Min moisture: {df_moist_plotted.min().min():.3f} m³/m³")
            st.write(f"Max moisture: {df_moist_plotted.max().max():.3f} m³/m³")
            st.markdown("---")

    # -----------------------------------------------------
    # Vertical Profile Explorer
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Vertical Profile Explorer")
    month_options = sorted(df_data.index.strftime("%Y-%m").unique())
    if month_options:
        chosen_month = st.selectbox("Choose Month–Year", month_options)
        prof_type = st.radio("Profile Type", ["Temperature", "Moisture"], horizontal=True)
        fig_prof = plot_vertical_profile(df_data, chosen_month, prof_type)
        if fig_prof:
            st.plotly_chart(fig_prof, use_container_width=False)
            # If temperature, let's see if any depth is below freezing
            if prof_type == "Temperature":
                sub_filtered = df_data[df_data.index.strftime("%Y-%m") == chosen_month]
                soil_temp_cols = [c for c in DEPTHS_TEMPERATURE.keys() if c in sub_filtered.columns]
                if not soil_temp_cols:
                    st.info("No soil temperature columns found for this month.")
                else:
                    # Simple approach: check how many hours each soil layer was below 0
                    st.write("#### Soil Frost Info")
                    for c in soil_temp_cols:
                        hours_below_freeze = (sub_filtered[c] < 0).sum()
                        if hours_below_freeze > 0:
                            st.write(f"- **{c}** had {hours_below_freeze} hour(s) below 0°C in {chosen_month}.")
                        else:
                            st.write(f"- **{c}** did not go below 0°C in {chosen_month}.")

    # -----------------------------------------------------
    # Monthly Statistics
    # -----------------------------------------------------
    st.markdown("---")
    st.subheader("Monthly Statistics")
    monthly_stats = df_data.resample("MS").agg(["mean", "min", "max"])
    monthly_stats.columns = [f"{col}_{stat}" for col, stat in monthly_stats.columns]
    monthly_stats.reset_index(inplace=True)
    monthly_stats.rename(columns={"date": "Month"}, inplace=True)

    def nice_stat_name(col_stat):
        parts = col_stat.rsplit("_", 1)
        if len(parts) == 2:
            base, stat = parts
            return f"{base} ({stat.capitalize()})"
        return col_stat

    monthly_stats.columns = [nice_stat_name(c) for c in monthly_stats.columns]

    with st.expander("View Monthly Statistics Table", expanded=False):
        st.dataframe(monthly_stats)

    # -----------------------------------------------------
    # Download CSV
    # -----------------------------------------------------
    csv_data = df_data.to_csv().encode("utf-8")
    st.download_button(
        "Download Full Data as CSV",
        data=csv_data,
        file_name="weather_data.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # Footer
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>Developed by <b>Arash Hosseini</b></p>"
        "<p style='text-align: center; font-size: 0.9em;'>Data provided by Open-Meteo Archive API</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
