import streamlit as st
import joblib
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime

# Load the trained model, scaler, and polynomial features transformer
model = joblib.load("l.pkl")
scaler = joblib.load("scaler.pkl")
polynom = joblib.load("polynom.pkl")

# Session state initialization
if "history" not in st.session_state:
    st.session_state["history"] = []
    
if "map_points" not in st.session_state:
    st.session_state["map_points"] = []

# App layout
st.set_page_config(page_title="Water Potability Classification", page_icon="💧", layout="wide")
st.title("Water Potability Classification")
st.markdown("""This app helps classify water quality as **Potable** or **Non-Potable** using key parameters.""")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    section = st.radio("Go to Section", ["Overview", "Interactive Classification", "Data Visualization", "FAQ"])
    st.markdown("---")
    st.markdown("### WHO Standards")
    st.markdown("- **pH**: 6.5 - 8.5\n- **Hardness**: ≤ 500 mg/L\n- **Turbidity**: ≤ 5 NTU\n- **Sulfate**: ≤ 500 mg/L")
    st.markdown("---")
    st.markdown("### Learn More")
    st.markdown("- [WHO Standards](https://www.who.int)\n- [Water Safety Guide](https://www.epa.gov)")
    st.info("Ensure all data inputs are accurate for optimal results.")

def create_parameter_visualization(data, parameter):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data[parameter],
        mode='lines+markers',
        name=parameter
    ))
    fig.update_layout(
        title=f"{parameter} Over Time",
        xaxis_title="Timestamp",
        yaxis_title=parameter,
        height=400
    )
    return fig

def create_distribution_plot(data, parameter):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data[parameter],
        name=parameter,
        nbinsx=20,
        opacity=0.7
    ))
    fig.update_layout(
        title=f"{parameter} Distribution",
        xaxis_title=parameter,
        yaxis_title="Count",
        height=400
    )
    return fig

def create_map():
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
    for point in st.session_state["map_points"]:
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=f"""
            Result: {point['result']}<br>
            pH: {point['ph']}<br>
            Hardness: {point['hardness']}<br>
            Timestamp: {point['timestamp']}
            """,
            icon=folium.Icon(color="green" if point["result"] == "Potable" else "red")
        ).add_to(m)
    return m

if section == "Overview":
    st.subheader("Overview")
    st.markdown("""
    This application uses a machine learning model to classify water quality based on the following parameters:

    - **pH**: Measures acidity or alkalinity (6.5-8.5 is ideal)
    - **Hardness**: Indicates calcium and magnesium concentration (≤500 mg/L recommended)
    - **Solids**: Total dissolved solids in water
    - **Chloramines**: Used for disinfection (2-4 mg/L is typical)
    - **Sulfate**: Natural substance (≤500 mg/L recommended)
    - **Conductivity**: Water's ability to conduct electricity
    - **Organic Carbon**: Indicates organic pollutants
    - **Trihalomethanes**: Disinfection by-products
    - **Turbidity**: Water clarity (≤5 NTU recommended)

    ### Features:
    - **Real-time Classification**: Get immediate water quality assessment
    - **Geographic Tracking**: Optional location mapping
    - **Historical Analysis**: Track changes over time with interactive visualizations
    - **Expert Recommendations**: Receive targeted advice based on results
    """)

elif section == "Interactive Classification":
    st.subheader("Interactive Classification")
    with st.form("single_entry_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0, step=1.0)
            solids = st.number_input("Total Solids (mg/L)", min_value=0.0, value=200.0, step=1.0)
        with col2:
            chloramines = st.number_input("Chloramines (mg/L)", min_value=0.0, value=1.0, step=0.1)
            sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=30.0, step=1.0)
            conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=200.0, step=1.0)
        with col3:
            organic_carbon = st.number_input("Organic Carbon (mg/L)", min_value=0.0, value=10.0, step=0.1)
            trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=50.0, step=1.0)
            turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=1.0, step=0.1)

        # Latitude and Longitude input
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")

        submitted = st.form_submit_button("Classify Water Quality")

        if submitted:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = "Potable"  # Default value for result

            # Check for parameter thresholds
            if ph < 6.5 or ph > 8.5:
                st.error("Water Quality: **NOT POTABLE** (pH out of range)", icon="❌")
                result = "Non-Potable"
            elif hardness > 500:
                st.error("Water Quality: **NOT POTABLE** (Hardness exceeds limit)", icon="❌")
                result = "Non-Potable"
            elif sulfate > 500:
                st.error("Water Quality: **NOT POTABLE** (Sulfate exceeds limit)", icon="❌")
                result = "Non-Potable"
            elif turbidity > 5:
                st.error("Water Quality: **NOT POTABLE** (Turbidity exceeds limit)", icon="❌")
                result = "Non-Potable"
            else:
                # Use ML model for prediction
                input_data = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                       organic_carbon, trihalomethanes, turbidity]])
                input_transformed = polynom.transform(input_data)
                input_scaled = scaler.transform(input_transformed)
                prediction = model.predict(input_scaled)

                # Display prediction
                if prediction[0] == 1:
                    st.success("Water Quality: **POTABLE**", icon="✅")
                    result = "Potable"
                else:
                    st.error("Water Quality: **NOT POTABLE**", icon="❌")
                    result = "Non-Potable"

            st.session_state["history"].append({
                "timestamp": timestamp,
                "pH": ph,
                "Hardness": hardness,
                "Total Solids": solids,
                "Chloramines": chloramines,
                "Sulfate": sulfate,
                "Conductivity": conductivity,
                "Organic Carbon": organic_carbon,
                "Trihalomethanes": trihalomethanes,
                "Turbidity": turbidity,
                "Result": result
            })

            # Add to map points only if latitude and longitude are provided
            if latitude is not None and longitude is not None:
                st.session_state["map_points"].append({
                    "lat": latitude,
                    "lon": longitude,
                    "result": result,
                    "ph": ph,
                    "hardness": hardness,
                    "timestamp": timestamp
                })

                # Show map if latitude and longitude are provided
                st.subheader("Testing Location")
                m = create_map()
                st_folium(m, width=700, height=500)


elif section == "Data Visualization":
    st.subheader("Historical Data Analysis")
    
    if not st.session_state["history"]:
        st.warning("No historical data available. Please classify some water samples first.")
    else:
        history_df = pd.DataFrame(st.session_state["history"])
        
        # Summary statistics
        st.markdown("### Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            total_samples = len(history_df)
            potable_samples = len(history_df[history_df["Result"] == "Potable"])
            st.metric("Total Samples", total_samples)
        with col2:
            potable_percentage = (potable_samples / total_samples * 100)
            st.metric("Potable Samples", f"{potable_percentage:.1f}%")
        with col3:
            recent_result = history_df.iloc[-1]["Result"]
            st.metric("Latest Result", recent_result)

        # Interactive parameter analysis
        st.markdown("### Parameter Trends")
        
        # Add tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Distributions", "Historical Data", "Map View"])
        
        with tab1:
            parameter = st.selectbox(
                "Select Parameter to Visualize (Time Series)",
                ["pH", "Hardness", "Turbidity", "Sulfate", "Conductivity"]
            )
            st.plotly_chart(create_parameter_visualization(history_df, parameter))
            
        with tab2:
            parameter2 = st.selectbox(
                "Select Parameter to Visualize (Distribution)",
                ["pH", "Hardness", "Turbidity", "Sulfate", "Conductivity"]
            )
            st.plotly_chart(create_distribution_plot(history_df, parameter2))
            
        with tab3:
            st.markdown("### Historical Data Table")
            # Add search/filter functionality
            search_term = st.text_input("Search in results", "")
            if search_term:
                filtered_df = history_df[history_df.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)]
            else:
                filtered_df = history_df
            
            # Add sorting functionality
            sort_col = st.selectbox("Sort by", history_df.columns.tolist())
            sort_order = st.radio("Sort order", ["Ascending", "Descending"])
            sorted_df = filtered_df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))
            
            # Display paginated table
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
            page_number = st.number_input("Page", min_value=1, max_value=max(1, len(sorted_df)//page_size + 1), value=1)
            start_idx = (page_number - 1) * page_size
            end_idx = start_idx + page_size
            
            st.dataframe(sorted_df.iloc[start_idx:end_idx], use_container_width=True)
            
            # Download button for filtered data
            csv_filtered = sorted_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Filtered Data",
                csv_filtered,
                "filtered_water_quality_history.csv",
                "text/csv"
            )
            
        with tab4:
            if st.session_state["map_points"]:
                st.markdown("### Geographic Distribution")
                st.markdown("Map showing all testing locations and their results")
                m = create_map()
                st_folium(m, width=700, height=500)
            else:
                st.info("No geographical data available. Include location information during classification to see the map.")

        # Download complete historical data
        st.markdown("### Download Complete Dataset")
        csv = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Complete History",
            csv,
            "water_quality_history.csv",
            "text/csv"
        )
elif section == "FAQ":
    st.subheader("Frequently Asked Questions")
    
    faq_data = {
        "What makes water potable?": """
        Potable water must meet specific quality standards:
        - p H between 6.5 and 8.5
        - Hardness below 500 mg/L
        - Turbidity under 5 NTU
        - Acceptable levels of minerals and chemicals
        - Free from harmful bacteria and contaminants
        """,
        
        "How often should I test my water?": """
        Testing frequency depends on your water source:
        - Municipal water: Annually
        - Private well: Every 6-12 months
        - After natural disasters or repairs: Immediately
        - If you notice changes in taste, odor, or color
        """,
        
        "What do the parameters mean?": """
        Key parameters explained:
        - pH: Measures acidity/alkalinity (ideal: 6.5-8.5)
        - Hardness: Calcium/magnesium content
        - Turbidity: Water clarity
        - Conductivity: Dissolved minerals
        - Organic Carbon: Organic matter content
        """,
        
        "How accurate is this tool?": """
        This tool provides an initial assessment based on water parameters, but:
        - Should not replace laboratory testing
        - Works best with accurate input data
        - Consider it a screening tool
        - Follow up with certified testing for critical applications
        """
    }
    
    for question, answer in faq_data.items():
        with st.expander(question):
            st.markdown(answer) 