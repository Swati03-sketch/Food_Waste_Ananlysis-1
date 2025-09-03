import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

def prepare_series_safe(df : pd.DataFrame, country = None, category = None):
    d = df.copy()
    if country : d = d[d['country'] == country]
    if category : d = d[d['food_category'] == category]
    s = d.groupby('date')['total_waste_(tons)'].sum().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(s.index):
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("MS")
    return s
def fit_forecast(series : pd.Series, periods=12, model_path=None):
    #Fit or load ARIMA model and forecast
    model_fit = None

    # Try loading existing model
    if model_path and os.path.exists(model_path):
        print(f"Loading ARIMA model from {model_path}")
        model_fit = ARIMAResults.load(model_path)
    else:
        print("Training new ARIMA model...")
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()

        # Save model for future runs
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_fit.save(model_path)
            print(f"💾 ARIMA model saved to {model_path}")
    
    fc = model_fit.forecast(steps = periods)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS") 
    forecast = pd.Series(fc.values, index=idx, name='forecast_waste_tons')
    return model_fit, forecast

def fallback_exponential_smoothing(series: pd.Series, periods=12, model_path=None):
    #Fit Exponential Smoothing and forecast
    print("Training Exponential Smoothing model...")
    model = ExponentialSmoothing(series, trend="add", seasonal=None)
    fit = model.fit()

    #save params manually later if needed
    fc = fit.forecast(periods)
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    forecast = pd.Series(fc.values, index=idx, name="forecast_waste_tons")
    return fit, forecast

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR,"data","processed","food_waste_clean.csv")
    output_path = os.path.join(BASE_DIR,"outputs","forecast_results.csv") 
    model_path = os.path.join(BASE_DIR,"outputs","models","global_arima.pkl")

    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    s = prepare_series_safe(df, country=None, category=None).dropna()
    print("Prepared series preview:\n", s.head(), "\nSeries length:", len(s))

    forecast = None
    method = None
    try:
        fit, forecast = fit_forecast(s, periods=6, model_path=model_path)
        if forecast.isna().all():
            raise ValueError("ARIMA produced only NaNs")
        method = "ARIMA"
        print("Forecast generated using ARIMA.")
    except Exception as e:
        print("ARIMA failed:", str(e))
        fit, forecast = fallback_exponential_smoothing(s, periods=6)
        method = "Exponential Smoothing"
        print("Forecast generated using Exponential Smoothing.")

    if forecast is None or forecast.nunique() == 1:
        model = ExponentialSmoothing(s.dropna(), trend="add", seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(6)
        method = "Exponential Smoothing"
        print("Exponential Smoothing forecast generated.")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df = pd.DataFrame({
        "date": forecast.index,
        f"{method}_forecast": forecast.values
    })
    forecast.to_csv(output_path, index=False)
    print(f"Forecast saved at: {output_path}")

    plt.figure(figsize=(8,5))
    s.plot(label="History")
    forecast.plot(label=f"{method} Forecast", marker="o")
    plt.legend()
    plt.title("Food Waste Forecast")
    plt.savefig(os.path.join(BASE_DIR, "outputs", "forecast_plot.png"))
    plt.show()