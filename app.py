from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Page configuration
st.set_page_config(
    page_title="Ticker Performance & DCA Comparator (No Pandas)",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for clean UI styling
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Sidebar UI components
st.sidebar.header("📊 Configuration Panel")

# 1. Ticker inputs
default_tickers = "ndq.ax, mnrs.ax, semi.ax"
tickers_input = st.sidebar.text_input(
    "Tickers (comma separated)", value=default_tickers
)

# 2. Date range inputs
default_end_date = datetime.today().date()
default_start_date = default_end_date - timedelta(days=180)  # ~6 months ago

start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date)

# 3. DCA option switch
enable_dca = st.sidebar.checkbox("Enable Daily DCA Simulation", value=False)
dca_amount = 0.0
if enable_dca:
  dca_amount = st.sidebar.number_input(
      "Daily DCA Amount ($)", min_value=1.0, value=10.0, step=5.0
  )

# 4. Run Button
run_button = st.sidebar.button("Run Simulation")

# Main Page Header
st.title("📈 Multi-Ticker Performance & DCA Comparator (Lightweight / No Pandas)")
st.markdown(
    "Compare relative percentage returns using pure NumPy and native Python arrays."
)


# Function to fetch and process data using raw dictionaries and numpy (No Pandas)
@st.cache_data(ttl=3600)
def load_data_native(tickers_str, start, end):
  tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
  if not tickers:
    return [], [], {}

  # Download via yfinance
  # Setting auto_adjust=True helps get clean Close prices directly
  df_yf = yf.download(
      tickers, start=start, end=end, progress=False, auto_adjust=True
  )

  if df_yf.empty:
    return [], [], {}

  # Extract dates
  dates = [d.strftime("%Y-%m-%d") for d in df_yf.index]
  price_data = {}

  # Handle single vs multi ticker column structures in yfinance output safely without pandas dataframe logic
  for ticker in tickers:
    try:
      if len(tickers) == 1:
        # Single ticker series
        if "Close" in df_yf.columns:
          series = df_yf["Close"].values
        else:
          series = df_yf.iloc[:, 0].values
      else:
        # Multi-ticker columns check
        if hasattr(df_yf.columns, "levels") and "Close" in df_yf.columns.levels[
            0
        ]:
          series = df_yf[("Close", ticker)].values
        elif ticker in df_yf.columns:
          series = df_yf[ticker].values
        else:
          # Fallback position matching
          idx = tickers.index(ticker)
          series = df_yf.iloc[:, idx].values

      # Clean NaN values into numpy array mask
      price_data[ticker] = np.array(series, dtype=float)
    except Exception:
      continue

  return dates, tickers, price_data


# Session state initialization
if "submitted" not in st.session_state:
  st.session_state.submitted = False

if run_button:
  st.session_state.submitted = True

# Main Execution Flow
if st.session_state.submitted:
  with st.spinner("Fetching market data via lightweight numpy arrays..."):
    dates, valid_tickers, price_data = load_data_native(
        tickers_input, start_date, end_date
    )

    if not dates or not price_data:
      st.error("No valid price data found. Please check your ticker symbols.")
    else:
      # Align and filter out days where any ticker has NaN values (inner join equivalent using numpy)
      valid_tickers = [t for t in valid_tickers if t in price_data]

      if not valid_tickers:
        st.error("No valid tickers matched data.")
      else:
        # Stack arrays to filter rows with NaNs
        stacked_prices = np.column_stack(
            [price_data[t] for t in valid_tickers]
        )
        not_nan_mask = ~np.isnan(stacked_prices).any(axis=1)

        filtered_dates = [
            d for i, d in enumerate(dates) if not_nan_mask[i]
        ]
        filtered_prices = stacked_prices[not_nan_mask]

        if len(filtered_dates) == 0:
          st.error("Insufficient overlapping price data after cleaning.")
        else:
          fig = go.Figure()
          color_palette = [
              "#2980b9",
              "#e74c3c",
              "#27ae60",
              "#8e44ad",
              "#f39c12",
              "#16a085",
          ]

          summary_data = []

          for idx, ticker in enumerate(valid_tickers):
            color = color_palette[idx % len(color_palette)]
            series = filtered_prices[:, idx]

            init_price = series[0]
            final_price = series[-1]
            total_price_return = (
                (final_price - init_price) / init_price
            ) * 100.0

            if enable_dca:
              # Pure NumPy DCA calculation
              shares_bought = dca_amount / series
              cumulative_shares = np.cumsum(shares_bought)
              cumulative_invested = dca_amount * np.arange(
                  1, len(series) + 1, dtype=float
              )
              portfolio_value = cumulative_shares * series
              multiplier = portfolio_value / cumulative_invested
              y_vals_pct = (multiplier - 1.0) * 100.0
              name_suffix = f" (DCA ${dca_amount}/day)"

              total_shares = np.sum(shares_bought)
              total_invested = dca_amount * len(series)
              final_value = total_shares * final_price
              dca_total_return = (
                  (final_value - total_invested) / total_invested
              ) * 100.0

              summary_data.append({
                  "Ticker": ticker,
                  "Start Price": f"{init_price:.2f}",
                  "End Price": f"{final_price:.2f}",
                  "Price Return (%)": f"{total_price_return:.2f}%",
                  "DCA Total Invested ($)": f"{total_invested:.2f}",
                  "DCA Final Value ($)": f"{final_value:.2f}",
                  "DCA Return (%)": f"{dca_total_return:.2f}%",
              })
            else:
              multiplier = series / init_price
              y_vals_pct = (multiplier - 1.0) * 100.0
              name_suffix = " (Price Return)"

              summary_data.append({
                  "Ticker": ticker,
                  "Start Price": f"{init_price:.2f}",
                  "End Price": f"{final_price:.2f}",
                  "Price Return (%)": f"{total_price_return:.2f}%",
              })

            # Add trace to Plotly
            fig.add_trace(go.Scatter(
                x=filtered_dates,
                y=multiplier,
                mode="lines",
                name=f"{ticker}{name_suffix}",
                line=dict(width=2, color=color),
                customdata=y_vals_pct,
                hovertemplate=(
                    "<b>%{text}</b><br>Date:"
                    " %{x}<br>Return: %{customdata:.2f}%<extra></extra>"
                ),
                text=[ticker] * len(filtered_dates),
            ))

          title_text = (
              f"DCA Strategy Return (%) - Daily ${dca_amount} (Log Scale)"
              if enable_dca
              else (
                  "Relative Percentage Return (%) [Normalized to 0% at Start]"
                  " (Log Scale)"
              )
          )

          fig.update_layout(
              title=title_text,
              xaxis_title="Date",
              yaxis=dict(
                  type="log",
                  title="Return Multiplier / Log Scale (0% = 1.0)",
                  tickvals=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
                  ticktext=[
                      "-75%",
                      "-50%",
                      "-25%",
                      "0%",
                      "+50%",
                      "+100%",
                      "+200%",
                      "+400%",
                      "+900%",
                  ],
              ),
              hovermode="x unified",
              template="plotly_white",
              legend=dict(
                  orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
              ),
              margin=dict(l=40, r=40, t=60, b=40),
          )

          st.plotly_chart(fig, use_container_width=True)

          # Summary Table rendered cleanly using native dictionaries / list of dicts (Streamlit natively supports list of dicts without Pandas!)
          st.subheader("📋 Performance Summary")
          st.dataframe(summary_data, use_container_width=True)
else:
  st.info(
      "👈 Please configure your parameters in the left sidebar and click"
      " **Run Simulation** to generate the chart."
  )
