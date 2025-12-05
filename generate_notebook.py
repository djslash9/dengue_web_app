import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dengue Prediction Model Training & Analysis (Thesis Version)\n",
    "This notebook covers the end-to-end modeling process for the Dengue Early Warning System.\n",
    "It includes:\n",
    "1.  **Exploratory Data Analysis (EDA)**: Distributions, Seasonality, Correlations.\n",
    "2.  **Statistical Modeling**: SARIMA (Baseline).\n",
    "3.  **Machine Learning**: XGBoost (Advanced).\n",
    "4.  **Evaluation**: Regression Metrics (RMSE, R²) and Classification Metrics (ROC for Outbreak Detection).\n",
    "5.  **Validation**: Time-Series Split Strategy."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc, confusion_matrix, classification_report\n",
    "import pickle\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from statsmodels.tsa.seasonal import seasonal_decompose\n",
    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n",
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.style.use('seaborn-v0_8-whitegrid')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading & Overview"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Data\n",
    "try:\n",
    "    df_awareness = pd.read_csv('awareness_score.csv')\n",
    "    df_dengue = pd.read_csv('dengue.csv')\n",
    "    \n",
    "    # Preprocess Date\n",
    "    df_dengue['Date'] = pd.to_datetime(df_dengue['Year'].astype(str) + '-' + df_dengue['Month No'].astype(str) + '-01')\n",
    "    \n",
    "    print(\"Data Loaded Successfully\")\n",
    "except FileNotFoundError:\n",
    "    print(\"Error: Data files not found.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Exploratory Data Analysis (EDA) for Thesis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.1 Distribution of Key Variables\n",
    "if 'df_dengue' in locals():\n",
    "    fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    "    \n",
    "    sns.histplot(df_dengue['Cases'], kde=True, ax=axes[0, 0], color='skyblue')\n",
    "    axes[0, 0].set_title('Distribution of Dengue Cases')\n",
    "    \n",
    "    sns.histplot(df_dengue['RainFall'], kde=True, ax=axes[0, 1], color='green')\n",
    "    axes[0, 1].set_title('Distribution of Rainfall')\n",
    "    \n",
    "    sns.histplot(df_dengue['MinTemp'], kde=True, ax=axes[1, 0], color='orange')\n",
    "    axes[1, 0].set_title('Distribution of Min Temp')\n",
    "    \n",
    "    sns.histplot(df_dengue['MaxTemp'], kde=True, ax=axes[1, 1], color='red')\n",
    "    axes[1, 1].set_title('Distribution of Max Temp')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2.2 Seasonal Decomposition (Colombo District)\n",
    "if 'df_dengue' in locals():\n",
    "    colombo_df = df_dengue[df_dengue['District'] == 'Colombo'].set_index('Date').sort_index()\n",
    "    \n",
    "    decomposition = seasonal_decompose(colombo_df['Cases'], model='additive', period=12)\n",
    "    fig = decomposition.plot()\n",
    "    fig.set_size_inches(12, 8)\n",
    "    plt.suptitle('Seasonal Decomposition of Dengue Cases (Colombo)', y=1.02)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Statistical Modeling: SARIMA (Baseline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3.1 SARIMA Implementation for Colombo\n",
    "if 'colombo_df' in locals():\n",
    "    # Split Train/Test\n",
    "    train_size = int(len(colombo_df) * 0.8)\n",
    "    train_sarima, test_sarima = colombo_df['Cases'][:train_size], colombo_df['Cases'][train_size:]\n",
    "    \n",
    "    # Fit SARIMA (Order selection would typically be automated, using (1,1,1)x(1,1,1,12) as example)\n",
    "    print(\"Training SARIMA Model...\")\n",
    "    sarima_model = SARIMAX(train_sarima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))\n",
    "    sarima_result = sarima_model.fit(disp=False)\n",
    "    \n",
    "    # Forecast\n",
    "    sarima_pred = sarima_result.get_forecast(steps=len(test_sarima))\n",
    "    sarima_pred_mean = sarima_pred.predicted_mean\n",
    "    sarima_conf_int = sarima_pred.conf_int()\n",
    "    \n",
    "    # Plot\n",
    "    plt.figure(figsize=(12, 6))\n",
    "    plt.plot(train_sarima.index, train_sarima, label='Train')\n",
    "    plt.plot(test_sarima.index, test_sarima, label='Test')\n",
    "    plt.plot(test_sarima.index, sarima_pred_mean, label='SARIMA Forecast', color='red', linestyle='--')\n",
    "    plt.fill_between(test_sarima.index, sarima_conf_int.iloc[:, 0], sarima_conf_int.iloc[:, 1], color='pink', alpha=0.3)\n",
    "    plt.title('SARIMA Forecast vs Actuals (Colombo)')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "    \n",
    "    # Evaluation\n",
    "    sarima_rmse = np.sqrt(mean_squared_error(test_sarima, sarima_pred_mean))\n",
    "    sarima_r2 = r2_score(test_sarima, sarima_pred_mean)\n",
    "    print(f\"SARIMA Performance - RMSE: {sarima_rmse:.2f}, R2: {sarima_r2:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Machine Learning: XGBoost (Advanced)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4.1 Feature Engineering (Same as before)\n",
    "if 'df_dengue' in locals():\n",
    "    # Calculate CDRI\n",
    "    cdri_ref = df_awareness.groupby('district')['total_score'].mean().reset_index()\n",
    "    cdri_ref.rename(columns={'total_score': 'CDRI', 'district': 'District'}, inplace=True)\n",
    "    \n",
    "    # Merge & Lag\n",
    "    df_merged = pd.merge(df_dengue, cdri_ref, on='District', how='left')\n",
    "    df_merged.sort_values(by=['District', 'Date'], inplace=True)\n",
    "    \n",
    "    lag_features = ['Cases', 'RainFall', 'MinTemp', 'MaxTemp']\n",
    "    for feature in lag_features:\n",
    "        df_merged[f'{feature}_Lag1'] = df_merged.groupby('District')[feature].shift(1)\n",
    "        \n",
    "    df_final = df_merged.dropna().reset_index(drop=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4.2 Train XGBoost\n",
    "features = ['Cases_Lag1', 'RainFall_Lag1', 'MinTemp_Lag1', 'MaxTemp_Lag1', 'CDRI', 'Population', 'Month No']\n",
    "target = 'Cases'\n",
    "\n",
    "X = df_final[features]\n",
    "y = df_final[target]\n",
    "\n",
    "# Chronological Split\n",
    "df_sorted_time = df_final[df_final['Date'].dt.year >= 2014].sort_values(by='Date').set_index('Date')\n",
    "X_sorted = df_sorted_time[features]\n",
    "y_sorted = df_sorted_time[target]\n",
    "\n",
    "split_index = int(len(df_sorted_time) * 0.8)\n",
    "X_train, X_test = X_sorted.iloc[:split_index], X_sorted.iloc[split_index:]\n",
    "y_train, y_test = y_sorted.iloc[:split_index], y_sorted.iloc[split_index:]\n",
    "\n",
    "model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "y_pred = model.predict(X_test)\n",
    "\n",
    "xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "xgb_r2 = r2_score(y_test, y_pred)\n",
    "print(f\"XGBoost Performance - RMSE: {xgb_rmse:.2f}, R2: {xgb_r2:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4.3 Validation Strategy Visualization\n",
    "plt.figure(figsize=(12, 4))\n",
    "plt.plot(y_train.index, y_train, label='Training Data')\n",
    "plt.plot(y_test.index, y_test, label='Testing Data')\n",
    "plt.title('Time Series Validation Split Strategy')\n",
    "plt.xlabel('Time Index')\n",
    "plt.ylabel('Cases')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Evaluation & Comparison"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5.1 Residual Analysis (XGBoost)\n",
    "residuals = y_test - y_pred\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "sns.histplot(residuals, kde=True, ax=axes[0], color='purple')\n",
    "axes[0].set_title('Distribution of Residuals')\n",
    "\n",
    "axes[1].scatter(y_pred, residuals, alpha=0.5)\n",
    "axes[1].axhline(0, color='red', linestyle='--')\n",
    "axes[1].set_xlabel('Predicted')\n",
    "axes[1].set_ylabel('Residuals')\n",
    "axes[1].set_title('Residuals vs Fitted')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5.2 Outbreak Detection (ROC Curve)\n",
    "# Define Outbreak: Cases > 75th percentile of historical data\n",
    "threshold = np.percentile(y, 75)\n",
    "y_test_binary = (y_test > threshold).astype(int)\n",
    "y_pred_binary = (y_pred > threshold).astype(int)\n",
    "\n",
    "# Calculate ROC\n",
    "fpr, tpr, _ = roc_curve(y_test_binary, y_pred)\n",
    "roc_auc = auc(fpr, tpr)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')\n",
    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('ROC Curve for Outbreak Detection')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5.3 Confusion Matrix for Outbreaks\n",
    "cm = confusion_matrix(y_test_binary, y_pred_binary)\n",
    "plt.figure(figsize=(6, 5))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Outbreak'], yticklabels=['Normal', 'Outbreak'])\n",
    "plt.title('Confusion Matrix (Outbreak Detection)')\n",
    "plt.ylabel('Actual')\n",
    "plt.xlabel('Predicted')\n",
    "plt.show()\n",
    "\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test_binary, y_pred_binary))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 5.4 Save Final Artifacts\n",
    "with open('dengue_model.pkl', 'wb') as f:\n",
    "    pickle.dump(model, f)\n",
    "with open('cdri_ref.pkl', 'wb') as f:\n",
    "    pickle.dump(cdri_ref, f)\n",
    "print(\"Final artifacts saved.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('model.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)
