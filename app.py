import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Ticker Performance & DCA Comparator",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .main {
        background-color: #faf8f5;
    }
    .stSidebar {
        background-color: #f4f1ea;
        padding: 20px;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        width: 100%;
        background-color: #2c3e50;
        color: white;
        font-weight: bold;
        border-radius: 4px;
        height: 45px;
    }
    .stButton>button:hover {
        background-color: #34495e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI components
st.sidebar.header("📊 Configuration Panel")

# 1. Ticker inputs
default_tickers = "ndq.ax, mnrs.ax, semi.ax"
tickers_input = st.sidebar.text_input("Tickers (comma separated)", value=default_tickers)

# 2. Date range inputs
default_end_date = datetime.today().date()
default_start_date = default_end_date - timedelta(days=180)  # ~6 months ago

start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date)

# 3. DCA option switch
enable_dca = st.sidebar.checkbox("Enable Daily DCA Simulation", value=False)
dca_amount = 0.0
if enable_dca:
    dca_amount = st.sidebar.number_input("Daily DCA Amount ($)", min_value=1.0, value=10.0, step=5.0)

# 4. Run Button (Prevents auto-rerun clutter on input change)
run_button = st.sidebar.button("Run Simulation")

# Main Page Header
st.title("📈 Multi-Ticker Performance & DCA Comparator")
st.markdown("Compare relative percentage returns and evaluate automated Dollar-Cost Averaging (DCA) strategies across selected assets.")

# Function to fetch and process data
@st.cache_data(ttl=3600)
def load_data(tickers_str, start, end):
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return pd.DataFrame(), []
    
    # Download data via yfinance
    df = yf.download(tickers, start=start, end=end, progress=False)
    
    if df.empty:
        return pd.DataFrame(), tickers
    
    # Handle single vs multiple tickers multi-index columns in yfinance
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.levels[0]:
            prices = df['Close']
        elif 'Adj Close' in df.columns.levels[0]:
            prices = df['Adj Close']
        else:
            prices = df.iloc[:, 0:len(tickers)]
    else:
        if 'Close' in df.columns:
            prices = df[['Close']]
            prices.columns = tickers
        else:
            prices = df
            
    # Drop rows where all values are NaN
    prices = prices.dropna(how='all')
    return prices, tickers

# Session state initialization to handle button click trigger
if "submitted" not in st.session_state:
    st.session_state.submitted = False

if run_button:
    st.session_state.submitted = True

# Main Execution Flow
if st.session_state.submitted:
    with st.spinner("Fetching market data and running simulations..."):
        prices_df, valid_tickers = load_data(tickers_input, start_date, end_date)
        
        if prices_df.empty or len(valid_tickers) == 0:
            st.error("No valid price data found for the given tickers and date range. Please check your inputs.")
        else:
            # Clean and align prices
            prices_df = prices_df.ffill().dropna()
            
            if prices_df.empty:
                st.error("Insufficient overlapping price data after cleaning.")
            else:
                fig = go.Figure()
                
                # Colors for consistency
                color_palette = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad', '#f39c12', '#16a085']
                
                for idx, ticker in enumerate(prices_df.columns):
                    color = color_palette[idx % len(color_palette)]
                    series = prices_df[ticker]
                    
                    if enable_dca:
                        # DCA Daily Simulation Calculation
                        shares_bought = dca_amount / series
                        cumulative_shares = shares_bought.cumsum()
                        cumulative_invested = dca_amount * np.arange(1, len(series) + 1)
                        portfolio_value = cumulative_shares * series
                        multiplier = portfolio_value / cumulative_invested
                        y_vals_pct = (multiplier - 1.0) * 100.0
                        name_suffix = f" (DCA ${dca_amount}/day)"
                    else:
                        base_price = series.iloc[0]
                        multiplier = series / base_price
                        y_vals_pct = (multiplier - 1.0) * 100.0
                        name_suffix = " (Price Return)"
                        
                    fig.add_trace(go.Scatter(
                        x=prices_df.index,
                        y=multiplier, # Using multiplier for clean log scale representation where 1.0 = 0%
                        mode='lines',
                        name=f"{ticker}{name_suffix}",
                        line=dict(width=2, color=color),
                        customdata=y_vals_pct,
                        hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Return: %{customdata:.2f}%<extra></extra>',
                        text=[ticker]*len(prices_df.index)
                    ))
                
                # Layout formatting with Log scale mapped to percentage multipliers
                title_text = f"DCA Strategy Return (%) - Daily ${dca_amount} (Log Scale)" if enable_dca else "Relative Percentage Return (%) [Normalized to 0% at Start] (Log Scale)"
                
                fig.update_layout(
                    title=title_text,
                    xaxis_title="Date",
                    yaxis=dict(
                        type="log",
                        title="Return Multiplier / Log Scale (0% = 1.0)",
                        tickvals=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                        ticktext=["-75%", "-50%", "-25%", "0%", "+50%", "+100%", "+200%", "+400%", "+900%"]
                    ),
                    hovermode="x unified",
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary Statistics Table
                st.subheader("📋 Performance Summary")
                summary_data = []
                for ticker in prices_df.columns:
                    series = prices_df[ticker]
                    final_price = series.iloc[-1]
                    init_price = series.iloc[0]
                    total_price_return = ((final_price - init_price) / init_price) * 100.0
                    
                    row = {
                        "Ticker": ticker,
                        "Start Price": f"{init_price:.2f}",
                        "End Price": f"{final_price:.2f}",
                        "Price Return (%)": f"{total_price_return:.2f}%"
                    }
                    if enable_dca:
                        shares_bought = dca_amount / series
                        total_shares = shares_bought.sum()
                        total_invested = dca_amount * len(series)
                        final_value = total_shares * final_price
                        dca_total_return = ((final_value - total_invested) / total_invested) * 100.0
                        row["DCA Total Invested ($)"] = f"{total_invested:.2f}"
                        row["DCA Final Value ($)"] = f"{final_value:.2f}"
                        row["DCA Return (%)"] = f"{dca_total_return:.2f}%"
                        
                    summary_data.append(row)
                    
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
else:
    st.info("👈 Please configure your parameters in the left sidebar and click **Run Simulation** to generate the chart.")
