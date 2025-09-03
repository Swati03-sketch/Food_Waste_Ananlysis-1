import pandas as pd
import os

def kpi_tables(df, output_file):
    os.makedirs("outputs", exist_ok=True)

    kpi_overall = pd.DataFrame({
        'total_waste_(tons)' : [df['total_waste_(tons)'].sum()],
        'total_economic_loss_(million_$)' : [df['economic_loss_(million_$)'].sum()],
        'avg_household_waste_(%)' : [df['household_waste_(%)'].mean()],
        'avg_per_capita_waste_kg' : [df['per_capita_waste_kg'].mean()],
    })

    by_year = (
        df.groupby('year', as_index=False)
        .agg({
            'total_waste_(tons)': 'sum',
              'economic_loss_(million_$)': 'sum',
              'per_capita_waste_kg': 'mean',
              'household_waste_(%)': 'mean',
        })
        .sort_values('year')
    )

    by_category = (
        df.groupby('food_category', as_index = False)
        .agg({
            'total_waste_(tons)': 'sum',
            'economic_loss_(million_$)': 'sum',
        })
        .sort_values('total_waste_(tons)', ascending = False)
    )

    by_country = (
        df.groupby('country',as_index = False)
        .agg({
            'total_waste_(tons)': 'sum',
            'household_waste_(%)': 'mean',
            'per_capita_waste_kg': 'mean',
            'economic_loss_(million_$)': 'sum',
            'population_(million)': 'max',
        })
    )

    with pd.ExcelWriter(output_file,engine='openpyxl') as writer:
        kpi_overall.to_excel(writer,sheet_name="KPI_Overall", index=False)
        by_year.to_excel(writer,sheet_name="By_Year",index=False)
        by_category.to_excel(writer, sheet_name="By_Category", index=False)
        by_country.to_excel(writer, sheet_name="By_Country", index=False)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
    input_file = os.path.join(BASE_DIR, "data", "processed", "food_waste_clean.csv")
    output_file = os.path.join(BASE_DIR, "outputs", "food_waste_analysis.xlsx")

    df = pd.read_csv(input_file, parse_dates=["date"])
    kpi_tables(df, output_file)