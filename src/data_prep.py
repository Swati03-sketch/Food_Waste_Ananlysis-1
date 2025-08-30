import pandas as pd
import numpy as np
import os

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # normalize column names
    df.columns = (df.columns
                  .str.strip().str.lower()
                  .str.replace(' ', '_'))
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # fix negative/implausible values
    for col in ['total_waste_(tons)', 'economic_loss_(million_$)', 
                'avg_waste_per_capita_(kg)', 'population_(million)']:
        if col in df:
            df.loc[df[col] < 0, col] = np.nan
    
    # cap extreme outliers
    for col in ['total_waste_(tons)', 'economic_loss_(million_$)']:
        if col in df:
            q99 = df[col].quantile(0.99)
            df[col] = np.where(df[col] > q99, q99, df[col])
    
    # consistent dtypes
    if 'year' in df:
        df['year'] = df['year'].astype(int)
    if 'population_(million)' in df:
        df['population_(million)'] = df['population_(million)'].astype(float)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Safe divisions
    if 'total_waste_(tons)' in df and 'population_(million)' in df:
        df['per_capita_waste_kg'] = (df['total_waste_(tons)'] * 1000) / (df['population_(million)'] * 1e6)

    if 'economic_loss_(million_$)' in df:
        df['economic_loss_per_ton'] = df['economic_loss_(million_$)'] / df['total_waste_(tons)']
    
    # Time fields
    if 'year' in df:
        df['date'] = pd.to_datetime(df['year'].astype(str) + "-01-01")

    return df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("C:/Users/Sony/Documents/global_food_wastage_dataset.csv")
    
    # Clean + feature engineering
    df_clean = basic_clean(df)
    df_feature = add_features(df_clean)

    print(df_feature.head())
    print("\nShape :", df.shape)

    # Ensure processed dir exists
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned dataset
    output_path = os.path.join(output_dir, "food_waste_clean.csv")
    df_feature.to_csv(output_path, index=False)

    print("Cleaned data saved to:", output_path)
    print("Shape:", df_feature.shape)
