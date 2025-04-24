import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title='EKEDC Meter Dashboard', layout='wide')

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('../daily_meter_summary.csv', parse_dates=['date_'])
    return df

# Add this at the top
if st.sidebar.button("🔄 Reload Data"):
    st.cache_data.clear()
df = load_data()

# ---------------------- FILTER SIDEBAR ----------------------
st.sidebar.header("🔎 Filter Data")

# Date Range Filter
date_range = st.sidebar.date_input("Select Date Range", value=[df['date_'].min(), df['date_'].max()])
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
filtered_df = df[(df['date_'] >= start_date) & (df['date_'] <= end_date)].copy()

# Band Filter with "Select All"
all_bands = ["Select All"] + sorted(filtered_df['band_'].dropna().unique().tolist())
selected_bands = st.sidebar.multiselect("Filter by Band", options=all_bands, default=["Select All"])
if "Select All" not in selected_bands:
    filtered_df = filtered_df[filtered_df['band_'].isin(selected_bands)]

# Location Filter with "Select All" (dependent on Band selection)
all_locations = ["Select All"] + sorted(filtered_df['location_id_'].dropna().unique().tolist())
selected_locations = st.sidebar.multiselect("Filter by Location", options=all_locations, default=["Select All"])
if "Select All" not in selected_locations:
    filtered_df = filtered_df[filtered_df['location_id_'].isin(selected_locations)]

# Meter Filter with "Select All" (dependent on Band and Location selection)
all_meters = ["Select All"] + sorted(filtered_df['meter_id_'].dropna().unique().tolist())
selected_meters = st.sidebar.multiselect("Filter by Meter ID", options=all_meters, default=["Select All"])
if "Select All" not in selected_meters:
    filtered_df = filtered_df[filtered_df['meter_id_'].isin(selected_meters)]

# Anomaly Filter
anomaly_options = sorted(filtered_df['anomaly_category'].unique().tolist())
selected_anomalies = st.sidebar.multiselect("Filter by Anomaly", options=anomaly_options, default=anomaly_options)
filtered_df = filtered_df[filtered_df['anomaly_category'].isin(selected_anomalies)]

st.title("⚡ EKEDC Daily Smart Meter Dashboard")

# ---------------------- KPI Formatter ----------------------
def format_metric(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:,.0f}"

# ---------------------- KPI Summary ----------------------
st.header("📊 Key Metrics Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown("### 💵 Total Vended")
    st.success(format_metric(filtered_df['total_vended_amount_sum'].sum()))

with kpi2:
    st.markdown("### 🔻 Revenue Loss")
    st.error(format_metric(filtered_df['revenue_loss_sum'].sum()))

with kpi3:
    st.markdown("### ⚡ Energy Loss (kWh)")
    st.warning(format_metric(filtered_df['energy_loss_kwh_sum'].sum()))

with kpi4:
    st.markdown("### 📈 Efficiency (%)")
    if not filtered_df.empty:
        st.info(f"{filtered_df['energy_efficiency_ratio'].mean() * 100:.2f}%")
    else:
        st.info("0.00%")

# ---------------------- EFFICIENCY & TRENDING METERS ----------------------
st.header("📈 Efficiency Trends and Top Performers")

col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        top_eff_meters = filtered_df.groupby("meter_id_")["energy_efficiency_ratio"].mean().nlargest(20).reset_index()
        fig = px.bar(top_eff_meters, x='meter_id_', y='energy_efficiency_ratio', title="Top 20 Efficient Meters", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display for Top 20 Efficient Meters based on current filters.")

with col2:
    if not filtered_df.empty:
        # Ensure 'top_eff_meters' is defined and not empty before filtering 'filtered_df'
        if 'top_eff_meters' in locals() and not top_eff_meters.empty:
            trend_df = filtered_df[filtered_df['meter_id_'].isin(top_eff_meters['meter_id_'])]
            if not trend_df.empty:
                fig2 = px.line(trend_df, x='date_', y='energy_efficiency_ratio', color='meter_id_', title="Efficiency Over Time")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No trend data available for the top efficient meters based on current filters.")
        else:
            st.info("No top efficient meters identified based on current filters.")
    else:
        st.info("No data to display for Efficiency Over Time based on current filters.")

# ---------------------- LOCATION ANALYSIS ----------------------
st.header("📍 Location-Based Performance")
if not filtered_df.empty:
    loc_group = filtered_df.groupby("location_id_").agg({
        "total_vended_amount_sum": "mean",
        "revenue_loss_sum": "mean",
        "energy_loss_kwh_sum": "mean",
        "energy_efficiency_ratio": "mean",
        "anomaly_severity_score": "mean"
    }).reset_index()

    fig_loc = px.bar(loc_group, x="location_id_", y=["total_vended_amount_sum", "revenue_loss_sum", "energy_loss_kwh_sum"], barmode="group", title="Avg KPIs by Location")
    st.plotly_chart(fig_loc, use_container_width=True)

    fig_eff_loc = px.bar(loc_group, x="location_id_", y="energy_efficiency_ratio", color="anomaly_severity_score", title="Efficiency & Severity by Location")
    st.plotly_chart(fig_eff_loc, use_container_width=True)
else:
    st.info("No location-based performance data to display based on current filters.")

# ---------------------- FEEDER & TRANSFORMER ANALYSIS ----------------------
st.header("🔌 Feeder and Transformer Health Overview")
col3, col4 = st.columns(2)

if "feeder_id_" in filtered_df.columns and not filtered_df.empty:
    feeder_group = filtered_df.groupby("feeder_id_").agg({
        "total_vended_amount_sum": "mean",
        "revenue_loss_sum": "mean",
        "energy_efficiency_ratio": "mean"
    }).reset_index().sort_values(by="total_vended_amount_sum", ascending=False).head(15)

    with col3:
        fig_feed = px.bar(feeder_group, x="feeder_id_", y="total_vended_amount_sum", color="energy_efficiency_ratio", title="Top 15 Feeders by Vended Amount")
        st.plotly_chart(fig_feed, use_container_width=True)
else:
    with col3:
        st.info("Feeder ID data not available or no data based on current filters.")

if "transformer_id_" in filtered_df.columns and not filtered_df.empty:
    tx_group = filtered_df.groupby("transformer_id_").agg({
        "total_vended_amount_sum": "mean",
        "revenue_loss_sum": "mean",
        "energy_efficiency_ratio": "mean"
    }).reset_index().sort_values(by="total_vended_amount_sum", ascending=False).head(15)

    with col4:
        fig_tx = px.bar(tx_group, x="transformer_id_", y="total_vended_amount_sum", color="energy_efficiency_ratio", title="Top 15 Transformers by Vended Amount")
        st.plotly_chart(fig_tx, use_container_width=True)
else:
    with col4:
        st.info("Transformer ID data not available or no data based on current filters.")

# ---------------------- ANOMALY DRILL-DOWN TABLE ----------------------
st.header("🧠 Anomaly Drill-Down Viewer")
if not filtered_df.empty:
    drill_df = filtered_df[[
        'meter_id_', 'date_', 'anomaly_category', 'anomaly_severity_score',
        'anomaly_summary_note', 'location_id_', 'feeder_id_', 'transformer_id_',
        'energy_efficiency_ratio', 'energy_loss_ratio', 'customer_risk_score_mean'
    ]]
    st.dataframe(drill_df)
else:
    st.info("No anomaly data to display based on current filters.")

# ---------------------- SMART ALERTS & SLA MONITORING ----------------------
st.header("🚨 Smart Alerts and SLA Monitoring")

if not filtered_df.empty:
    # Define anomaly subsets based on the filtered data
    high_loss = filtered_df[(filtered_df['revenue_loss_sum'] > 100000) & (filtered_df['is_industrial_'] == 1)]
    sla_fail = filtered_df[filtered_df['band_compliance_max'] == 0]
    burnt_out = filtered_df[filtered_df['fault_type_'] == 'burnt_meter']

    # Unique meter counts
    high_loss_count = high_loss['meter_id_'].nunique()
    sla_fail_count = sla_fail['meter_id_'].nunique()
    burnt_meter_count = burnt_out['meter_id_'].nunique()

    # Revenue losses
    high_loss_value = high_loss['revenue_loss_sum'].sum()
    sla_loss_value = sla_fail['revenue_loss_sum'].sum()
    burnt_loss_value = burnt_out['revenue_loss_sum'].sum()

    # Display alerts
    if high_loss_count > 0:
        st.error(f"🔴 High Revenue Loss in **{high_loss_count} unique industrial meters**, Loss: ₦{high_loss_value:,.0f}")
    if sla_fail_count > 0:
        st.warning(f"⚠️ SLA Violations in **{sla_fail_count} unique meters**, Loss: ₦{sla_loss_value:,.0f}")
    if burnt_meter_count > 0:
        st.error(f"🔥 Burnt Faults in **{burnt_meter_count} meters**, Loss: ₦{burnt_loss_value:,.0f}")
    else:
        st.info("No smart alerts based on the current filters.")
else:
    st.info("No data available for smart alerts based on current filters.")

# ---------------------- SLA COMPLIANCE BAR CHART ----------------------
st.markdown("### 📊 SLA Compliance Rate per Tariff Band")
if not filtered_df.empty and 'band_' in filtered_df.columns and 'band_compliance_max' in filtered_df.columns:
    sla_band = filtered_df.groupby("band_")["band_compliance_max"].mean().reset_index()
    sla_band["compliance_rate (%)"] = sla_band["band_compliance_max"] * 100

    fig_sla = px.bar(
        sla_band,
        x="band_",
        y="compliance_rate (%)",
        color="compliance_rate (%)",
        color_continuous_scale="Blues",
        title="SLA Compliance Rate by Tariff Band"
    )
    st.plotly_chart(fig_sla, use_container_width=True)
else:
    st.info("SLA Compliance Rate data not available or no data based on current filters.")

# ---------------------- SLA DELIVERY ANALYSIS ----------------------
st.markdown("### ⏳ SLA Supply Hours vs Expected")
if not filtered_df.empty and 'band_' in filtered_df.columns and 'supply_hours' in filtered_df.columns:
    sla_hours = {"A": 20, "B": 16, "C": 12, "D": 8, "E": 4}
    filtered_df["expected_sla_hours"] = filtered_df["band_"].map(sla_hours)

    sla_delivery = filtered_df.groupby("band_")[["supply_hours", "expected_sla_hours"]].mean().reset_index()
    if not sla_delivery.empty and 'expected_sla_hours' in sla_delivery.columns and sla_delivery['expected_sla_hours'].notna().all():
        sla_delivery["actual_vs_expected (%)"] = (sla_delivery["supply_hours"] / sla_delivery["expected_sla_hours"]) * 100

        fig_sla_hours = px.bar(
            sla_delivery,
            x="band_",
            y="actual_vs_expected (%)",
            title="Average SLA Supply vs Expected by Band",
            color="actual_vs_expected (%)",
            color_continuous_scale="Teal"
        )
        st.plotly_chart(fig_sla_hours, use_container_width=True)
    else:
        st.info("Expected SLA hours data is missing or incomplete for current filters.")
else:
    st.info("SLA Supply Hours data not available or no data based on current filters.")

# ---------------------- SLA COMPLIANCE TREND ----------------------
st.markdown("### 📅 SLA Compliance Trend Over Time")
if not filtered_df.empty and 'date_' in filtered_df.columns and 'band_' in filtered_df.columns and 'band_compliance_max' in filtered_df.columns:
    sla_trend = filtered_df.groupby(["date_", "band_"])["band_compliance_max"].mean().reset_index()
    sla_trend["compliance_rate (%)"] = sla_trend["band_compliance_max"] * 100

    fig_sla_trend = px.line(
        sla_trend,
        x="date_",
        y="compliance_rate (%)",
        color="band_",
        markers=True,
        title="Daily SLA Compliance Rate by Band"
    )
    st.plotly_chart(fig_sla_trend, use_container_width=True)
else:
    st.info("SLA Compliance Trend data not available or no data based on current filters.")

# ---------------------- Anomaly Subset Export ----------------------
st.markdown("### 📤 Export Anomaly Subsets")
if not filtered_df.empty:
    anomaly_option = st.selectbox("Choose anomaly subset to download", options=["High Loss", "SLA Failures", "Burnt Faults"])
    export_df = {
        "High Loss": filtered_df[(filtered_df['revenue_loss_sum'] > 100000) & (filtered_df['is_industrial_'] == 1)],
        "SLA Failures": filtered_df[filtered_df['band_compliance_max'] == 0],
        "Burnt Faults": filtered_df[filtered_df['fault_type_'] == 'burnt_meter']
    }[anomaly_option]

    csv_data = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download {anomaly_option} Data (CSV)",
        data=csv_data,
        file_name=f"{anomaly_option.lower().replace(' ', '_')}_anomalies.csv",
        mime="text/csv"
    )
else:
    st.info("No data available to export based on current filters.")

# ---------------------- MAP-BASED VISUALIZATION ----------------------
st.header("🗺️ Geographic Heatmap of Efficiency and Risk (Simulated Coordinates)")
if not filtered_df.empty and 'location_id_' in filtered_df.columns:
    # Simulate coordinates for demonstration
    np.random.seed(42)
    coords = filtered_df[['location_id_']].drop_duplicates().reset_index(drop=True)
    coords['lat'] = np.random.uniform(6.4, 6.7, size=len(coords))  # Lagos area
    coords['lon'] = np.random.uniform(3.2, 3.5, size=len(coords))
    df_coords = filtered_df.merge(coords, on='location_id_', how='left')

    map_fig = px.scatter_map(
        df_coords,
        lat="lat",
        lon="lon",
        color="anomaly_severity_score",
        size=df_coords["revenue_loss_sum"].abs() + 1 , # avoid 0 too,
        hover_name="location_id_",
        hover_data=["energy_efficiency_ratio", "anomaly_category"],
        color_continuous_scale="YlOrRd",
        size_max=25,
        zoom=10,
        title="Location Risk and Efficiency Map"
    )
    map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(map_fig, use_container_width=True)
else:
    st.info("Location data not available or no data based on current filters for the map.")

# ---------------------- METER-SPECIFIC TIME SERIES ----------------------
st.header("⏳ Meter Drill-Down Time Series")
if not filtered_df.empty and 'meter_id_' in filtered_df.columns and 'date_' in filtered_df.columns and 'energy_efficiency_ratio' in filtered_df.columns and 'energy_loss_ratio' in filtered_df.columns and 'revenue_loss_sum' in filtered_df.columns:
    selected_meter = st.selectbox("Select a Meter ID to Drill Down", options=filtered_df['meter_id_'].unique())
    meter_data = filtered_df[filtered_df['meter_id_'] == selected_meter].sort_values('date_')

    fig_mtr = px.line(
        meter_data,
        x='date_',
        y=['energy_efficiency_ratio', 'energy_loss_ratio', 'revenue_loss_sum'],
        title=f"Efficiency, Loss & Revenue Over Time for {selected_meter}"
    )
    st.plotly_chart(fig_mtr, use_container_width=True)
else:
    st.info("No meter data available for the time series chart based on current filters.")

# ---------------------- EXPORT TOOLS ----------------------
st.header("📤 Data Export & Summary Reports")

if not filtered_df.empty:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data (CSV)", csv, "filtered_data.csv", "text/csv")

    html_report = filtered_df.describe().to_html()
    st.download_button("📥 Download Summary Report (HTML)", html_report, "summary_report.html", "text/html")
else:
    st.info("No filtered data available for download.")