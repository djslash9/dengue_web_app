import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import xgboost as xgb
import numpy as np
from datetime import datetime, timedelta

# Page Config
st.set_page_config(page_title="DEWRS - Dengue Early Warning System", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Menu Styling */
    .stRadio > label {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        padding-bottom: 10px;
    }
    .stRadio div[role='radiogroup'] > label {
        font-size: 1.1rem !important;
        padding: 10px 10px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .stRadio div[role='radiogroup'] > label:hover {
        background-color: #f0f2f6;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        color: #d32f2f;
    }
    
    /* Chart Box Border */
    .chart-box {
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        background-color: #ffffff;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: #555;
        text-align: center;
        padding: 10px;
        font-size: 0.9rem;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        with open('dengue_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('cdri_ref.pkl', 'rb') as f:
            cdri_ref = pickle.load(f)
        return model, cdri_ref
    except FileNotFoundError:
        st.error("Artifacts not found. Please run dengue.py first.")
        return None, None

@st.cache_data
def load_data():
    try:
        dengue_df = pd.read_csv('dengue.csv')
        awareness_df = pd.read_csv('awareness_score.csv')
        # Preprocess Dates
        dengue_df['Date'] = pd.to_datetime(dengue_df['Year'].astype(str) + '-' + dengue_df['Month No'].astype(str) + '-01')
        return dengue_df, awareness_df
    except FileNotFoundError:
        st.error("Data files not found.")
        return None, None

model, cdri_ref = load_artifacts()
dengue_df, awareness_df = load_data()

if model is not None and dengue_df is not None:
    
    # --- Sidebar Navigation ---
    st.sidebar.title("DEWRS Navigation")
    
    # Menu Options
    menu_options = ["Dashboard", "Historical Analysis", "Survey Insights", "Future Forecasts", "Recommendations", "Datasets"]
    selection = st.sidebar.radio("Go to", menu_options, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Dengue Early Warning & Response System v5.0")

    # --- MAIN CONTENT ---
    
    # --- VIEW: Dashboard ---
    if selection == "Dashboard":
        st.title("DEWRS Dashboard")
        col_dash_1, col_dash_2 = st.columns([1, 2])
        
        with col_dash_1:
            st.subheader("Prediction Engine")
            st.markdown("Enter last month's conditions:")

            districts = ['Colombo', 'Gampaha', 'Kalutara']
            selected_district = st.selectbox("Select District", districts, key="dash_district")

            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0, key="dash_rain")
            min_temp = st.number_input("Min Temp (°C)", min_value=0.0, value=25.0, key="dash_min")
            max_temp = st.number_input("Max Temp (°C)", min_value=0.0, value=30.0, key="dash_max")
            cases_lag1 = st.number_input("Last Month Cases", min_value=0, value=50, key="dash_cases")
            month_no = st.slider("Prediction Month No", 1, 12, 1, key="dash_month")

            predict_btn = st.button("Predict Risk", key="dash_btn")

        with col_dash_2:
            if predict_btn:
                # Get CDRI
                district_cdri_row = cdri_ref[cdri_ref['District'] == selected_district]
                cdri_val = district_cdri_row['CDRI'].values[0] if not district_cdri_row.empty else 50.0
                
                # Get Population (Latest)
                pop_row = dengue_df[dengue_df['District'] == selected_district].sort_values(by='Date', ascending=False).head(1)
                population = pop_row['Population'].values[0] if not pop_row.empty else 100000

                # Prepare input
                input_data = pd.DataFrame({
                    'Cases_Lag1': [cases_lag1],
                    'RainFall_Lag1': [rainfall],
                    'MinTemp_Lag1': [min_temp],
                    'MaxTemp_Lag1': [max_temp],
                    'CDRI': [cdri_val],
                    'Population': [population],
                    'Month No': [month_no]
                })

                # Predict
                prediction = model.predict(input_data)[0]
                predicted_cases = int(max(0, prediction))

                # Display Result
                st.metric(label=f"Predicted Cases for {selected_district}", value=f"{predicted_cases}")

                # Action Authority Logic
                st.subheader("Recommended Response")
                risk_level = "High" if predicted_cases > 1000 else "Low"
                awareness_level = "High" if cdri_val >= 65 else "Low"

                if risk_level == "High" and awareness_level == "Low":
                    st.error("🚨 EMERGENCY STATUS: Immediate legal enforcement of breeding sites & fogging required. Community compliance is low.")
                elif risk_level == "High" and awareness_level == "High":
                    st.warning("⚠️ ALERT: Mobilize community cleanup campaigns. Public is receptive but environmental factors are critical.")
                else:
                    st.success("✅ SURVEILLANCE: Maintain routine monitoring.")
            else:
                st.info("Select a district and click 'Predict Risk' to see recommendations.")

    # --- VIEW: Historical Analysis ---
    elif selection == "Historical Analysis":
        st.title("Historical Data Analysis")
        
        districts = ['Colombo', 'Gampaha', 'Kalutara']
        
        # Top-level Filters
        col_filters_1, col_filters_2 = st.columns(2)
        with col_filters_1:
            hist_district = st.selectbox("Filter by District", ['All'] + districts, key="hist_dist")
        with col_filters_2:
            years = sorted(dengue_df['Year'].unique())
            selected_years = st.multiselect("Select Years", years, default=years, key="hist_years")
        
        # Filter Data
        hist_df = dengue_df[dengue_df['Year'].isin(selected_years)]
        if hist_district != 'All':
            hist_df = hist_df[hist_df['District'] == hist_district]

        st.markdown("---")

        # Visuals in Single Column (Vertically Stacked)
        
        # 1. Cases Over Time
        st.subheader("Dengue Cases Trend")
        fig_cases = px.line(hist_df, x='Date', y='Cases', color='District', title='Dengue Cases Over Time')
        st.plotly_chart(fig_cases, use_container_width=True)

        # 2. Climate Factors
        st.subheader("Climate Factors")
        fig_climate = go.Figure()
        if hist_district == 'All':
            clim_agg = hist_df.groupby('Date')[['RainFall', 'MinTemp', 'MaxTemp']].mean().reset_index()
        else:
            clim_agg = hist_df
        
        fig_climate.add_trace(go.Bar(x=clim_agg['Date'], y=clim_agg['RainFall'], name='Rainfall', marker_color='blue', opacity=0.5))
        fig_climate.add_trace(go.Scatter(x=clim_agg['Date'], y=clim_agg['MaxTemp'], name='Max Temp', yaxis='y2', line=dict(color='orange')))
        fig_climate.add_trace(go.Scatter(x=clim_agg['Date'], y=clim_agg['MinTemp'], name='Min Temp', yaxis='y2', line=dict(color='green')))
        
        fig_climate.update_layout(
            yaxis=dict(title='Rainfall (mm)'),
            yaxis2=dict(title='Temp (°C)', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_climate, use_container_width=True)

        # 3. Population Trends
        st.subheader("Population Trend")
        fig_pop = px.line(hist_df, x='Date', y='Population', color='District', title='Population Growth Over Time')
        fig_pop.update_layout(yaxis=dict(tickformat="d")) # Format as integer (full number)
        st.plotly_chart(fig_pop, use_container_width=True)

        # 4. Incidence Rate
        st.subheader("Incidence Rate (Cases vs Population)")
        hist_df['Incidence_Rate'] = (hist_df['Cases'] / hist_df['Population']) * 100
        fig_inc = px.line(hist_df, x='Date', y='Incidence_Rate', color='District', title='Dengue Incidence Rate (%)')
        fig_inc.update_layout(yaxis_title="Incidence Rate (%)")
        st.plotly_chart(fig_inc, use_container_width=True)

        # 5. Year-on-Year Comparison (Using Global Filters)
        st.subheader("Year-on-Year Seasonality")
        # Use hist_df which is already filtered by Year and District
        if hist_district == 'All':
            yoy_agg = hist_df.groupby(['Year', 'Month No'])['Cases'].sum().reset_index()
            title_yoy = "Yearly Seasonality (All Districts - Sum of Cases)"
        else:
            yoy_agg = hist_df
            title_yoy = f"Yearly Seasonality in {hist_district}"
        
        fig_yoy = px.line(yoy_agg, x='Month No', y='Cases', color='Year', title=title_yoy)
        st.plotly_chart(fig_yoy, use_container_width=True)

    # --- VIEW: Survey Insights ---
    elif selection == "Survey Insights":
        st.title("Community Survey Insights")
        districts = ['Colombo', 'Gampaha', 'Kalutara']
        
        if awareness_df is not None:
            col_surv_1, col_surv_2 = st.columns(2)
            
            with col_surv_1:
                # Age Distribution
                fig_age = px.pie(awareness_df, names='age', title='Respondent Age Distribution')
                st.plotly_chart(fig_age, use_container_width=True)
                
                # Occupation
                fig_occ = px.bar(awareness_df['occupation'].value_counts().reset_index(), x='occupation', y='count', title='Occupation Distribution')
                st.plotly_chart(fig_occ, use_container_width=True)

            with col_surv_2:
                # Gender
                fig_gender = px.pie(awareness_df, names='gender', title='Gender Distribution')
                st.plotly_chart(fig_gender, use_container_width=True)
                
                # Education
                fig_edu = px.bar(awareness_df['education'].value_counts().reset_index(), x='education', y='count', title='Education Levels')
                st.plotly_chart(fig_edu, use_container_width=True)
            
            # Radar Chart (Enhanced)
            st.subheader("Vulnerability Profile (Scores)")
            radar_cols = ['knowledge_score', 'attitudes_score', 'preventive_score', 'total_score']
            radar_df = awareness_df.groupby('district')[radar_cols].mean().reset_index()
            categories = radar_cols
            
            fig_radar = go.Figure()
            for district in districts:
                values = radar_df[radar_df['district'] == district][categories].values.flatten().tolist()
                values += values[:1]
                # Enhanced Trace: Lines + Markers + Text
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, 
                    theta=categories + [categories[0]], 
                    fill='toself', 
                    name=district,
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" for v in values],
                    textposition="top center",
                    textfont=dict(size=12, color="black") # Clearer numbers
                ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                title="Average Awareness Scores by District",
                height=700, # Bigger size
                font=dict(size=14)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Awareness Score Table
            st.subheader("District Awareness Scores")
            st.dataframe(radar_df.style.background_gradient(cmap='Blues'), use_container_width=True)

    # --- VIEW: Future Forecasts ---
    elif selection == "Future Forecasts":
        st.title("Future Forecasts (Next 12 Months)")
        districts = ['Colombo', 'Gampaha', 'Kalutara']
        
        forecast_district = st.selectbox("Select District for Forecast", ['All'] + districts, key="fc_dist")
        
        if st.button("Generate Forecast", key="fc_btn"):
            
            districts_to_forecast = districts if forecast_district == 'All' else [forecast_district]
            all_forecasts = []

            for dist in districts_to_forecast:
                # Logic per district
                last_row = dengue_df[dengue_df['District'] == dist].sort_values(by='Date').iloc[-1]
                district_hist = dengue_df[dengue_df['District'] == dist]
                monthly_avgs = district_hist.groupby('Month No')[['RainFall', 'MinTemp', 'MaxTemp']].mean()
                
                district_cdri_row = cdri_ref[cdri_ref['District'] == dist]
                cdri_val = district_cdri_row['CDRI'].values[0] if not district_cdri_row.empty else 50.0
                population = last_row['Population'] 

                future_preds = []
                start_date = last_row['Date'] + pd.DateOffset(months=1)
                
                for i in range(12):
                    future_date = start_date + pd.DateOffset(months=i)
                    month_idx = future_date.month
                    
                    rain_est = monthly_avgs.loc[month_idx, 'RainFall']
                    min_est = monthly_avgs.loc[month_idx, 'MinTemp']
                    max_est = monthly_avgs.loc[month_idx, 'MaxTemp']
                    
                    if i == 0:
                        prev_cases = last_row['Cases']
                        prev_rain = last_row['RainFall']
                        prev_min = last_row['MinTemp']
                        prev_max = last_row['MaxTemp']
                    else:
                        prev_cases = future_preds[-1]['Predicted_Cases']
                        prev_date = future_date - pd.DateOffset(months=1)
                        prev_month_idx = prev_date.month
                        prev_rain = monthly_avgs.loc[prev_month_idx, 'RainFall']
                        prev_min = monthly_avgs.loc[prev_month_idx, 'MinTemp']
                        prev_max = monthly_avgs.loc[prev_month_idx, 'MaxTemp']

                    input_data = pd.DataFrame({
                        'Cases_Lag1': [prev_cases],
                        'RainFall_Lag1': [prev_rain],
                        'MinTemp_Lag1': [prev_min],
                        'MaxTemp_Lag1': [prev_max],
                        'CDRI': [cdri_val],
                        'Population': [population],
                        'Month No': [month_idx]
                    })
                    
                    pred = model.predict(input_data)[0]
                    pred_cases = int(max(0, pred))
                    
                    future_preds.append({
                        'Date': future_date,
                        'Predicted_Cases': pred_cases,
                        'District': dist,
                        'Type': 'Forecast'
                    })
                
                all_forecasts.extend(future_preds)

            forecast_df = pd.DataFrame(all_forecasts)
            
            # --- 1. High Risk Analysis ---
            st.subheader("⚠️ High Risk Analysis (> 1000 Cases)")
            high_risk_df = forecast_df[forecast_df['Predicted_Cases'] > 1000].sort_values(by='Predicted_Cases', ascending=False)
            
            if not high_risk_df.empty:
                st.error(f"CRITICAL ALERT: {len(high_risk_df)} high-risk instances detected.")
                st.dataframe(high_risk_df[['Date', 'District', 'Predicted_Cases']].style.background_gradient(cmap='Reds'))
                
                st.markdown("### 🛡️ Recommended Precautions for High Risk Areas")
                st.markdown("""
                *   **Immediate Fogging**: Deploy teams to identified high-risk districts 2 weeks prior to the predicted month.
                *   **Public Awareness**: Launch targeted SMS campaigns in these specific areas.
                *   **Hospital Prep**: Increase bed capacity and stock IV fluids in local hospitals.
                *   **Legal Action**: Enforce strict penalties for uncleaned premises in these zones.
                """)
            else:
                st.success("No high-risk months predicted (> 1000 cases). Maintain routine surveillance.")

            st.markdown("---")

            # --- 2. Visualizations (Comparison) ---
            st.subheader("📊 Forecast vs Historical Trends")
            
            # Prepare Historical Data for Comparison
            if forecast_district == 'All':
                hist_comp_df = dengue_df.copy()
            else:
                hist_comp_df = dengue_df[dengue_df['District'] == forecast_district].copy()
            
            hist_comp_df['Type'] = 'Actual'
            hist_comp_df.rename(columns={'Cases': 'Predicted_Cases'}, inplace=True) # Rename for merging
            
            # Combine Data
            combined_df = pd.concat([hist_comp_df[['Date', 'District', 'Predicted_Cases', 'Type']], forecast_df[['Date', 'District', 'Predicted_Cases', 'Type']]])
            
            # Chart 1: Last 12 Months + Forecast
            st.markdown("#### Short-Term Outlook (Last 12 Months + Forecast)")
            last_date = dengue_df['Date'].max()
            start_12m = last_date - pd.DateOffset(months=12)
            short_term_df = combined_df[combined_df['Date'] > start_12m]
            
            fig_short = px.line(short_term_df, x='Date', y='Predicted_Cases', color='Type', line_dash='Type', 
                                title=f"Actual vs Forecast (Short Term) - {forecast_district}",
                                color_discrete_map={'Actual': 'blue', 'Forecast': 'red'})
            st.plotly_chart(fig_short, use_container_width=True)
            
            # Chart 2: Long-Term Trend
            st.markdown("#### Long-Term Trend (All History + Forecast)")
            fig_long = px.line(combined_df, x='Date', y='Predicted_Cases', color='Type', 
                               title=f"Long Term Trend - {forecast_district}",
                               color_discrete_map={'Actual': 'blue', 'Forecast': 'orange'})
            st.plotly_chart(fig_long, use_container_width=True)

            st.markdown("---")

            # --- 3. Monthly Trend Cards ---
            st.subheader("📉 Monthly Trend Analysis")
            
            # Aggregate if 'All' to show overall trend, or show per district? 
            # Let's show overall trend if 'All', or specific if District selected.
            if forecast_district == 'All':
                trend_df = forecast_df.groupby('Date')['Predicted_Cases'].sum().reset_index()
            else:
                trend_df = forecast_df
            
            cols = st.columns(4)
            for i, row in trend_df.iterrows():
                date_str = row['Date'].strftime("%b %Y")
                cases = row['Predicted_Cases']
                
                if i > 0:
                    prev_cases = trend_df.iloc[i-1]['Predicted_Cases']
                    diff = cases - prev_cases
                    delta_color = "inverse" # Red if up, Green if down
                else:
                    diff = 0
                    delta_color = "off"
                
                with cols[i % 4]:
                    st.metric(label=date_str, value=f"{cases}", delta=f"{diff}", delta_color=delta_color)

        st.markdown("---")

        # --- 4. Custom Prediction Tool ---
        with st.expander("🛠️ Custom Monthly Prediction Tool (What-If Analysis)"):
            st.markdown("Predict cases for a specific scenario.")
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                c_district = st.selectbox("District", districts, key="cust_dist")
                c_month = st.slider("Month No", 1, 12, 1, key="cust_month")
                c_cases_lag = st.number_input("Previous Month Cases", value=100, key="cust_cases")
            
            with c_col2:
                c_rain = st.number_input("Rainfall (mm)", value=150.0, key="cust_rain")
                c_min_t = st.number_input("Min Temp (°C)", value=24.0, key="cust_min")
                c_max_t = st.number_input("Max Temp (°C)", value=31.0, key="cust_max")
            
            if st.button("Predict Custom Scenario"):
                # Get CDRI & Pop
                c_cdri_row = cdri_ref[cdri_ref['District'] == c_district]
                c_cdri = c_cdri_row['CDRI'].values[0] if not c_cdri_row.empty else 50.0
                c_pop_row = dengue_df[dengue_df['District'] == c_district].sort_values(by='Date').iloc[-1]
                c_pop = c_pop_row['Population']
                
                c_input = pd.DataFrame({
                    'Cases_Lag1': [c_cases_lag],
                    'RainFall_Lag1': [c_rain],
                    'MinTemp_Lag1': [c_min_t],
                    'MaxTemp_Lag1': [c_max_t],
                    'CDRI': [c_cdri],
                    'Population': [c_pop],
                    'Month No': [c_month]
                })
                
                c_pred = model.predict(c_input)[0]
                st.success(f"Predicted Cases for {c_district} in Month {c_month}: **{int(c_pred)}**")

    # --- VIEW: Recommendations ---
    elif selection == "Recommendations":
        st.title("Decision Maker Recommendations")
        
        st.markdown("""
        ### 🛡️ Strategic Interventions
        
        Based on the DEWRS analysis, the following actions are recommended for policy makers:
        
        #### 1. Pre-Monsoon Vector Control
        *   **Timing**: Initiate 1 month before peak rainfall months (typically May & October).
        *   **Action**: Deploy fogging teams and inspect high-risk breeding sites (construction sites, schools).
        
        #### 2. Community Engagement (Based on Survey)
        *   **Target**: Districts with low 'Knowledge' or 'Attitude' scores.
        *   **Action**: Launch awareness campaigns focusing on the specific gaps identified in the Survey Insights tab.
        *   **Medium**: Use social media for younger demographics (18-24) and community centers for older groups.
        
        #### 3. Emergency Response Protocol
        *   **Trigger**: When Predicted Cases > 1000 AND CDRI < 65.
        *   **Action**: Declare high-risk zone, enforce legal penalties for breeding sites, and mobilize rapid response teams.
        
        #### 4. Resource Allocation
        *   Use the **Future Forecasts** to stockpile medicines and IV fluids in hospitals 2 months in advance of predicted peaks.
        """)
        
        st.info("💡 **Tip**: Use the 'Prediction Engine' in the Dashboard tab to simulate different climate scenarios and see how they affect risk levels.")

        st.markdown("---")
        st.caption("Sources: These recommendations are based on technical guidelines from the **World Health Organization (WHO)** Global Strategy for Dengue Prevention and Control (2012-2020) and the **National Dengue Control Unit (NDCU)**, Ministry of Health, Sri Lanka.")

    # --- VIEW: Datasets ---
    elif selection == "Datasets":
        st.title("Dataset Viewer")
        
        st.subheader("Dengue Data (Climate & Cases)")
        st.dataframe(dengue_df)
        
        st.subheader("Community Awareness Survey Data")
        st.dataframe(awareness_df)

    # --- Footer ---
    st.markdown("""
        <div class="footer">
            <p>Created by Sisira Dharmasena, University of Kelaniya, 2025</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please ensure data and model artifacts are present.")
