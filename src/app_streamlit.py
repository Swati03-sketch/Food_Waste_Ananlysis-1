import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Food Waste Dashboard", layout="wide")
sns.set_style("whitegrid")

# Helper Functions
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "processed", "food_waste_clean.csv")
    forecast_path = os.path.join(base_dir, "outputs", "forecast_results.csv")
    clusters_path = os.path.join(base_dir, "outputs", "country_clusters.csv")

    df = pd.read_csv(data_path)
    forecast = (pd.read_csv(forecast_path, parse_dates=["date"])
                if os.path.exists(forecast_path) else pd.DataFrame())
    clusters = pd.read_csv(clusters_path) if os.path.exists(clusters_path) else pd.DataFrame()

    return df, forecast, clusters

def compute_kpis(df):
    kpi_overall = {
        'Total Waste (tons)': df['total_waste_(tons)'].sum(),
        'Total Economic Loss (million $)': df['economic_loss_(million_$)'].sum(),
        'Avg Household Waste (%)': df['household_waste_(%)'].mean(),
        'Avg Per Capita Waste (kg)': df['per_capita_waste_kg'].mean()
    }
    return kpi_overall


# Main App
st.title("Food Waste Analysis Dashboard")

# Load data
df, forecast_df, clusters_df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_country = st.sidebar.selectbox("Select Country", ["All"] + df['country'].unique().tolist())
selected_category = st.sidebar.selectbox("Select Food Category", ["All"] + df['food_category'].unique().tolist())

if selected_country != "All":
    df = df[df['country'] == selected_country]
if selected_category != "All":
    df = df[df['food_category'] == selected_category]

# KPI Metrics
st.header("📊 Key KPIs")
kpis = compute_kpis(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Waste (tons)", f"{kpis['Total Waste (tons)']:,}")
col2.metric("Total Economic Loss (M $)", f"{kpis['Total Economic Loss (million $)']:,}")
col3.metric("Avg Household Waste (%)", f"{kpis['Avg Household Waste (%)']:.2f}")
col4.metric("Avg Per Capita Waste (kg)", f"{kpis['Avg Per Capita Waste (kg)']:.2f}")

# Waste Trends by Year
st.header("📈 Waste Trends by Year")
by_year = df.groupby('year', as_index=False)['total_waste_(tons)'].sum()
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=by_year, x='year', y='total_waste_(tons)', marker="o", ax=ax)
ax.set_ylabel("Total Waste (tons)")
st.pyplot(fig)

# Waste by Category
st.header("🍽️ Waste by Food Category")
by_category = df.groupby('food_category', as_index=False)['total_waste_(tons)'].sum().sort_values('total_waste_(tons)', ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=by_category, x='food_category', y='total_waste_(tons)', palette="viridis", ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Forecast Plot
st.header("🔮 Forecasted Waste")
if not forecast_df.empty and 'ARIMA_forecast' in forecast_df.columns:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.lineplot(x=forecast_df['date'], y=forecast_df['ARIMA_forecast'], marker='o', ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Waste (tons)")
    st.pyplot(fig)
else:
    st.warning("Forecast data is empty or missing values!!")

# Country Clusters
st.header("🌍 Country Clusters")
if not clusters_df.empty:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=clusters_df, x='total_waste', y='economic_loss', hue='cluster', palette="deep", s=100, ax=ax)
    ax.set_xlabel("Total Waste (tons)")
    ax.set_ylabel("Economic Loss (M $)")
    st.pyplot(fig)

st.success("Dashboard Loaded Successfully!")
