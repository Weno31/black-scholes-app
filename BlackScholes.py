import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.stats import norm
import seaborn as  sns
import os 

st.set_page_config(page_title="Black Scholes Options Pricing", layout="wide", page_icon=":money_with_wings:")

# Black-Scholes pricing model
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def pnl_calc(option_price, purchase_price):
    return option_price - purchase_price

#Sidebar LinkedIn Profile Link
st.sidebar.markdown("""
    <a href="https://www.linkedin.com/in/golesedimonngakgotla/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25">
        <span style="vertical-align: middle; margin-left: 8px;">LinkedIn</span>
    </a>
""", unsafe_allow_html=True)

#Sidebar Inputs
st.sidebar.title("⚖️ Option Parameters")
S = st.sidebar.number_input("Spot Price (S)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T = st.sidebar.number_input("Time to Maturity (T in years)", value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (sigma)", value=0.2)
option_type = st.sidebar.selectbox("Option Type", ['call', 'put'])
purchase_price = st.sidebar.number_input("Purchase Price of Option", value=5.0)

#Sidebar HeatMap Inputs
st.sidebar.title(":thermometer: Heatmap Settings")
strike_range = st.sidebar.slider("Strike Range %", min_value=0.1, max_value=0.5, value=0.2)
num_strikes = st.sidebar.slider("Number of Strike Points", min_value=5, max_value=50, value=21)

# Calculate option price and PnL
option_price = black_scholes_price(S, K, T, r, sigma, option_type)
pnl = pnl_calc(option_price, purchase_price)

# Show main metrics
st.title(":chart_with_upwards_trend: Black-Scholes Option Calculator")
st.subheader("by Golesedi Monngakgotla")
col1, col2 = st.columns(2)
col1.metric(label="Option Price", value=f"${option_price:.2f}")
col2.metric(label="PnL", value=f"${pnl:.2f}")

# Generate option chain for heatmap
strikes = np.linspace(S * (1 - strike_range), S * (1 + strike_range), num_strikes)
spot_prices = np.linspace(S * (1 - strike_range), S * (1 + strike_range), num_strikes)

heatmap_data = []
data_for_csv = []

for s in spot_prices:
    row = []
    for k in strikes:
        price = black_scholes_price(s, k, T, r, sigma, option_type)
        this_pnl = pnl_calc(price, purchase_price)
        row.append(this_pnl)
        data_for_csv.append({
            "Spot Price": s,
            "Strike Price": k,
            "T": T,
            "r": r,
            "sigma": sigma,
            "Option Type": option_type,
            "Purchase Price": purchase_price,
            "Option Price": price,
            "PnL": this_pnl
        })
    heatmap_data.append(row)

# Create DataFrame for heatmap
heatmap_df = pd.DataFrame(heatmap_data, index=np.round(spot_prices, 2), columns=np.round(strikes, 2))

st.subheader(":fire: PnL Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_df, cmap="RdYlGn", annot=False, ax=ax)
ax.set_xlabel("Strike Price")
ax.set_ylabel("Spot Price")
st.pyplot(fig)

# Convert to CSV
csv_df = pd.DataFrame(data_for_csv)
csv_filename = "option_analysis.csv"

with st.expander("📘 How This App Works"):
    st.markdown("""
    **How the Web App Works:**

    1. **User Inputs**:  
       Enter the current stock price, strike price, time to maturity (in years), risk-free rate, volatility, option type (call or put), and your purchase price.

    2. **Price Calculation**:  
       The app uses the Black-Scholes model to compute the theoretical price of the option, and its sensitivities (known as the Greeks).

    3. **PnL Computation**:  
       Profit or Loss (PnL) is calculated as:  
       `PnL = Current Option Value - Purchase Price`

    4. **Heatmap Visualization**:  
       A heatmap shows how the PnL would change for different combinations of spot and strike prices.

    ---

    **What the Heatmap Colors Represent:**

    - Each cell shows the PnL for a particular spot and strike price.
    - 🟩 **Green** = Profit (darker = higher gain)  
    - 🟥 **Red** = Loss (darker = deeper loss)  
    - ⚪ **White/Gray** = Around breakeven

    This helps you visualize risk and reward under different scenarios quickly and intuitively.
    """)
