import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def calculate_cdri(awareness_path):
    """
    Calculates the Community Dengue Risk Index (CDRI) from awareness scores.
    """
    print("Loading awareness data...")
    try:
        df_awareness = pd.read_csv(awareness_path)
    except FileNotFoundError:
        print(f"Error: {awareness_path} not found.")
        return None

    # Group by district and calculate mean total_score
    print("Calculating CDRI...")
    cdri_ref = df_awareness.groupby('district')['total_score'].mean().reset_index()
    cdri_ref.rename(columns={'total_score': 'CDRI', 'district': 'District'}, inplace=True)
    
    return cdri_ref

def feature_engineering(dengue_path, cdri_ref):
    """
    Performs feature engineering on the dengue dataset.
    """
    print("Loading dengue data...")
    try:
        df_dengue = pd.read_csv(dengue_path)
    except FileNotFoundError:
        print(f"Error: {dengue_path} not found.")
        return None

    # Merge CDRI
    print("Merging CDRI...")
    # Ensure District names match (case-insensitive strip might be good practice, but assuming clean data for now)
    df_merged = pd.merge(df_dengue, cdri_ref, on='District', how='left')

    # Create Date column
    print("Creating Date column...")
    # Assuming 'Month' is a string name, we might need to convert it or use 'Month No'
    # 'Month No' is likely 1-12.
    df_merged['Date'] = pd.to_datetime(df_merged['Year'].astype(str) + '-' + df_merged['Month No'].astype(str) + '-01')

    # Sort by District and Date to ensure correct lagging
    df_merged.sort_values(by=['District', 'Date'], inplace=True)

    # Create Lag Features (t-1)
    print("Creating Lag features...")
    lag_features = ['Cases', 'RainFall', 'MinTemp', 'MaxTemp']
    for feature in lag_features:
        df_merged[f'{feature}_Lag1'] = df_merged.groupby('District')[feature].shift(1)

    # Drop rows with NaN values created by lagging (first month of each district will be NaN)
    # Also drop if CDRI is missing (though left merge might leave some if districts don't match)
    df_final = df_merged.dropna().reset_index(drop=True)
    
    return df_final

def train_model(df):
    """
    Trains the XGBoost model.
    """
    print("Training XGBoost model...")
    
    # Define features and target
    features = ['Cases_Lag1', 'RainFall_Lag1', 'MinTemp_Lag1', 'MaxTemp_Lag1', 'CDRI', 'Population', 'Month No']
    target = 'Cases'

    X = df[features]
    y = df[target]

    # Chronological split
    # Since we sorted by Date, we can just split by index or time. 
    # Let's use a simple time-based split (e.g., last 20% as test) to respect temporal order across all districts
    # However, simply splitting by index might mix districts if not careful.
    # Better approach for time series: Split based on Date threshold.
    # But for simplicity and general requirement, we'll use a non-shuffled train_test_split which effectively does a chronological split if data is sorted by time (which it is, primarily).
    # Wait, we sorted by District THEN Date. So simply splitting by index would take the last few districts as test set. That's BAD.
    # We need to split chronologically GLOBAL or per district.
    # Let's sort by Date globally for the split.
    
    df_sorted_time = df.sort_values(by='Date')
    X_sorted = df_sorted_time[features]
    y_sorted = df_sorted_time[target]

    # Split: 80% train, 20% test
    split_index = int(len(df_sorted_time) * 0.8)
    X_train, X_test = X_sorted.iloc[:split_index], X_sorted.iloc[split_index:]
    y_train, y_test = y_sorted.iloc[:split_index], y_sorted.iloc[split_index:]

    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # Train XGBoost Regressor
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return model

def save_artifacts(model, cdri_ref):
    """
    Saves the trained model and CDRI reference.
    """
    print("Saving artifacts...")
    with open('dengue_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('cdri_ref.pkl', 'wb') as f:
        pickle.dump(cdri_ref, f)
    
    print("Artifacts saved: dengue_model.pkl, cdri_ref.pkl")

def main():
    awareness_file = 'awareness_score.csv'
    dengue_file = 'dengue.csv'

    # 1. Calculate CDRI
    cdri_ref = calculate_cdri(awareness_file)
    if cdri_ref is None:
        return

    # 2. Feature Engineering
    df_final = feature_engineering(dengue_file, cdri_ref)
    if df_final is None:
        return

    if df_final.empty:
        print("Error: Processed dataframe is empty. Check data merging and lagging.")
        return

    # 3. Train Model
    model = train_model(df_final)

    # 4. Save Artifacts
    save_artifacts(model, cdri_ref)
    print("Process completed successfully.")

if __name__ == "__main__":
    main()
