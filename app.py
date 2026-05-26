import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(
    page_title="Stock Market Analysis",
    page_icon="📈",
    layout="wide"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
}
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.title("📈 Stock Market Analysis Dashboard")
st.markdown("Interactive stock analysis using Streamlit")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "csv"]
)

# -------------------------
# Load Data
# -------------------------
if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Date conversion
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

        df.sort_index(inplace=True)

        st.success("Dataset Loaded Successfully")

        # -------------------------
        # Dataset Preview
        # -------------------------
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # -------------------------
        # Metrics
        # -------------------------
        if 'Close' in df.columns:

            latest_close = round(df['Close'].iloc[-1], 2)
            highest_close = round(df['Close'].max(), 2)
            lowest_close = round(df['Close'].min(), 2)

            col1, col2, col3 = st.columns(3)

            col1.metric("Latest Close", latest_close)
            col2.metric("Highest Close", highest_close)
            col3.metric("Lowest Close", lowest_close)

            # -------------------------
            # Closing Price Chart
            # -------------------------
            st.subheader("Closing Price Trend")

            fig = px.line(
                df,
                x=df.index,
                y='Close',
                title='Stock Closing Price',
                template='plotly_dark'
            )

            st.plotly_chart(fig, use_container_width=True)

            # -------------------------
            # Moving Average
            # -------------------------
            st.subheader("Moving Average Analysis")

            short_window = st.slider(
                "Select Short Moving Average",
                5,
                50,
                20
            )

            long_window = st.slider(
                "Select Long Moving Average",
                50,
                200,
                100
            )

            df['MA_Short'] = df['Close'].rolling(short_window).mean()
            df['MA_Long'] = df['Close'].rolling(long_window).mean()

            fig2 = px.line(
                df,
                x=df.index,
                y=['Close', 'MA_Short', 'MA_Long'],
                template='plotly_dark',
                title='Moving Average Comparison'
            )

            st.plotly_chart(fig2, use_container_width=True)

            # -------------------------
            # Volume Analysis
            # -------------------------
            if 'Volume' in df.columns:

                st.subheader("Trading Volume")

                fig3 = px.bar(
                    df,
                    x=df.index,
                    y='Volume',
                    template='plotly_dark',
                    title='Volume Analysis'
                )

                st.plotly_chart(fig3, use_container_width=True)

            # -------------------------
            # Seasonal Decomposition
            # -------------------------
            st.subheader("Seasonal Decomposition")

            try:
                decomposition = seasonal_decompose(
                    df['Close'].dropna(),
                    model='multiplicative',
                    period=30
                )

                fig4, axes = plt.subplots(4, 1, figsize=(12, 10))

                decomposition.observed.plot(ax=axes[0], title='Observed')
                decomposition.trend.plot(ax=axes[1], title='Trend')
                decomposition.seasonal.plot(ax=axes[2], title='Seasonality')
                decomposition.resid.plot(ax=axes[3], title='Residual')

                plt.tight_layout()

                st.pyplot(fig4)

            except:
                st.warning("Not enough data for seasonal decomposition.")

            # -------------------------
            # Correlation Heatmap
            # -------------------------
            st.subheader("Correlation Heatmap")

            numeric_df = df.select_dtypes(include=np.number)

            if not numeric_df.empty:

                corr = numeric_df.corr()

                fig5 = px.imshow(
                    corr,
                    text_auto=True,
                    color_continuous_scale='Viridis',
                    template='plotly_dark'
                )

                st.plotly_chart(fig5, use_container_width=True)


            # -------------------------
            # Next 30 Days Prediction
            # -------------------------

            st.subheader("Next 30 Days Prediction")

            last_price = df['Close'].iloc[-1]

            predictions = []

            current_price = last_price

            # Generate Predictions
            for i in range(30):

                change_percent = np.random.normal(0, 0.02)

                current_price = current_price * (1 + change_percent)

                predictions.append(current_price)

            # Create Future Dates
            future_dates = pd.date_range(
                start=pd.Timestamp.today(),
                periods=30,
                freq='D'
            )

            # Create Prediction DataFrame
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions
            })

            # Show Prediction Table
            st.subheader("Predicted Prices")

            st.dataframe(forecast_df, use_container_width=True)

            # Forecast Chart
            fig_forecast = px.line(
                forecast_df,
                x='Date',
                y='Predicted Price',
                title='Next 30 Days Forecast',
                template='plotly_dark'
            )

            st.plotly_chart(fig_forecast, use_container_width=True)
            # -------------------------
            # Raw Data
            # -------------------------
            with st.expander("View Full Dataset"):
                st.dataframe(df, use_container_width=True)

        else:
            st.error("Dataset must contain a 'Close' column.")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

else:
    st.info("Upload your stock dataset to begin analysis.")
