import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def prepare_series_safe(df : pd.DataFrame, country = None, category = None):
    d = df.copy()
    if country : d = d[d['country'] == country]
    if category : d = d[d['food_category'] == category]
    s = d.groupby('date')['total_waste_(tons)'].sum().sort_index()
    if pd.api.types.is_datetime64_any_dtype(s.index):
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("MS")
    return s
def fit_forecast(series : pd.Series, periods=12):
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    fc = model_fit.forecast(steps = periods)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS") 
    forecast = pd.Series(fc, index=idx, name='forecast_waste_tons')
    return model_fit, forecast

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR,"data","processed","food_waste_clean.csv")
    output_path = os.path.join(BASE_DIR,"outputs","forecast_results.csv") 
    
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    s = df['total_waste_(tons)'].resample("MS").sum()

    forecast = None
    method = None
    try:
        model = ARIMA(s.dropna(), order=(1,1,1))
        fit = model.fit()
        forecast = fit.forecast(steps=6)
        method = "ARIMA"
        print("Forecast generates.")
        print(forecast.head())
    except Exception as e:
        print("Failed, switching to exponential smoothing.")

    if forecast is None or forecast.nunique() == 1:
        model = ExponentialSmoothing(s.dropna(), trend="add", seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(6)
        method = "Exponential Smoothing"
        print("Exponential Smoothing forecast generated.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast.to_csv(output_path, header=[f"{method}_forecast"])
    print(f"Forecast saved at: {output_path}")

    plt.figure(figsize=(8,5))
    s.plot(label="History")
    forecast.plot(label=f"{method} Forecast", marker="o")
    plt.legend()
    plt.title("Food Waste Forecast")
    plt.savefig(os.path.join(BASE_DIR, "outputs", "forecast_plot.png"))
    plt.show()