import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



LOOKBACK = 60

st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("Stock Price Predictor")

# Model cache
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}

# ── Helpers 

def fetch_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        return df
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


# ── Technical Indicator Functions 

def calculate_sma(prices, period):
    """
    Formula: SMA = (P1 + P2 + ... + Pn) / n
    Where P = Price, n = Period
    """
    sma = []
    for i in range(len(prices)):
        if i < period - 1:
            sma.append(np.nan)
        else:
            window_sum = sum(prices[i - period + 1:i + 1])
            sma.append(window_sum / period)
    return np.array(sma)


def calculate_ema(prices, period):
    """
    Formula: EMA = Price(t) x k + EMA(y) x (1 - k)
    Where k = 2 / (period + 1)
    """
    ema = []
    k = 2 / (period + 1)
    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[i])
        else:
            ema_value = (prices[i] * k) + (ema[i-1] * (1 - k))
            ema.append(ema_value)
    return np.array(ema)


def calculate_rsi(prices, period=14):
    """
    Formula: RSI = 100 - [100 / (1 + RS)]
    Where RS = Average Gain / Average Loss
    """
    rsi = []
    gains = []
    losses = []
    for i in range(len(prices)):
        if i == 0:
            gains.append(0)
            losses.append(0)
            rsi.append(np.nan)
        else:
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
            if i < period:
                rsi.append(np.nan)
            elif i == period:
                avg_gain = sum(gains[-period:]) / period
                avg_loss = sum(losses[-period:]) / period
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
            else:
                prev_avg_gain = (sum(gains[-period-1:-1]) / period)
                prev_avg_loss = (sum(losses[-period-1:-1]) / period)
                avg_gain = (prev_avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (prev_avg_loss * (period - 1) + losses[i]) / period
                if avg_loss == 0:
                    rsi.append(100)
                else:
                    rs = avg_gain / avg_loss
                    rsi.append(100 - (100 / (1 + rs)))
    return np.array(rsi)


def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Formula:
    MACD Line = EMA(12) - EMA(26)
    Signal Line = EMA(9) of MACD Line
    MACD Histogram = MACD Line - Signal Line
    """
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)
    macd_line = ema_fast - ema_slow
    macd_valid_start = slow_period - 1
    macd_for_signal = macd_line[macd_valid_start:]
    signal_line_values = calculate_ema(macd_for_signal, signal_period)
    signal_line = np.full(len(prices), np.nan)
    signal_line[macd_valid_start:] = signal_line_values
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram


def calculate_bollinger_bands(prices, period=20, num_std=2):
    """
    Formula:
    Middle Band = SMA(20)
    Upper Band = SMA(20) + (Standard Deviation x 2)
    Lower Band = SMA(20) - (Standard Deviation x 2)
    """
    sma = calculate_sma(prices, period)
    upper_band = []
    lower_band = []
    for i in range(len(prices)):
        if i < period - 1:
            upper_band.append(np.nan)
            lower_band.append(np.nan)
        else:
            window = prices[i - period + 1:i + 1]
            mean = sum(window) / period
            variance = sum((x - mean) ** 2 for x in window) / period
            std_dev = variance ** 0.5
            upper_band.append(sma[i] + (std_dev * num_std))
            lower_band.append(sma[i] - (std_dev * num_std))
    return sma, np.array(upper_band), np.array(lower_band)


def calculate_stochastic(high, low, close, period=14, smooth_k=3):
    """
    Formula:
    %K = [(Close - Lowest Low) / (Highest High - Lowest Low)] x 100
    %D = SMA of %K (3-period)
    """
    k_values = []
    for i in range(len(close)):
        if i < period - 1:
            k_values.append(np.nan)
        else:
            window_high  = high[i - period + 1:i + 1]
            window_low   = low[i  - period + 1:i + 1]
            highest_high = max(window_high)
            lowest_low   = min(window_low)
            if highest_high == lowest_low:
                k_values.append(50)
            else:
                k = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
                k_values.append(k)
    k_values = np.array(k_values)
    d_values = calculate_sma(k_values, smooth_k)
    return k_values, d_values


def calculate_atr(high, low, close, period=14):
    """
    Formula:
    True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    ATR = SMA of True Range over period
    """
    true_ranges = []
    for i in range(len(close)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i]  - close[i-1])
            tr = max(hl, hc, lc)
        true_ranges.append(tr)
    return calculate_sma(np.array(true_ranges), period)


def calculate_obv(close, volume):
    """
    Formula:
    If Close > Close(previous): OBV = OBV(previous) + Volume
    If Close < Close(previous): OBV = OBV(previous) - Volume
    If Close = Close(previous): OBV = OBV(previous)
    """
    obv = []
    for i in range(len(close)):
        if i == 0:
            obv.append(volume[i])
        else:
            if close[i] > close[i-1]:
                obv.append(obv[i-1] + volume[i])
            elif close[i] < close[i-1]:
                obv.append(obv[i-1] - volume[i])
            else:
                obv.append(obv[i-1])
    return np.array(obv)


def add_indicators(df):
    df = df.copy()
    prices = df['Close'].values
    high   = df['High'].values
    low    = df['Low'].values
    volume = df['Volume'].values

    # SMA
    df['SMA_20']  = calculate_sma(prices, 20)
    df['SMA_50']  = calculate_sma(prices, 50)
    df['SMA_200'] = calculate_sma(prices, 200)

    # EMA
    df['EMA_12'] = calculate_ema(prices, 12)
    df['EMA_26'] = calculate_ema(prices, 26)

    # RSI
    df['RSI'] = calculate_rsi(prices, 14)

    # MACD
    macd, signal, hist = calculate_macd(prices)
    df['MACD']        = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist']   = hist

    # Bollinger Bands
    bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(prices)
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BB_Mid']   = bb_mid

    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(high, low, prices)
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d

    # ATR
    df['ATR'] = calculate_atr(high, low, prices)

    # OBV
    df['OBV'] = calculate_obv(prices, volume)

    # Moving Averages for main chart
    df['MA_20']  = df['SMA_20']
    df['MA_50']  = df['SMA_50']
    df['MA_200'] = df['SMA_200']

    df.reset_index(drop=True, inplace=True)
    return df


def build_sequences(scaled_close, lookback):
    X, y = [], []
    for i in range(lookback, len(scaled_close)):
        X.append(scaled_close[i - lookback:i, 0])
        y.append(scaled_close[i, 0])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def build_model(input_shape):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    tf.keras.backend.clear_session()
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    selected_stock = st.text_input("Stock Symbol", value="AAPL",
                                   placeholder="e.g. AAPL, TSLA, NVDA...").upper().strip()

    st.markdown("**Start Date**")
    start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1),
                               min_value=datetime.date(2010, 1, 1),
                               max_value=datetime.date.today(),
                               label_visibility="collapsed")
    st.markdown("**End Date**")
    end_date = st.date_input("End Date", value=datetime.date.today(),
                             min_value=datetime.date(2010, 1, 1),
                             max_value=datetime.date.today(),
                             label_visibility="collapsed")

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    if not selected_stock:
        st.error("Please enter a stock symbol.")
        st.stop()

    cache_key = f"{selected_stock}_{start_date}"
    is_cached = cache_key in st.session_state.model_cache

    if is_cached:
        st.success(f"✅ Model for {selected_stock} already trained this session.")

    predict_btn = st.button(
        "Predict " if is_cached else "Train & Predict",
        type="primary", use_container_width=True
    )

# ── Fetch Data ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {selected_stock} data..."):
    try:
        full_data    = fetch_data(selected_stock, str(start_date),
                                  datetime.date.today().strftime('%Y-%m-%d'))
        display_data = fetch_data(selected_stock, str(start_date), str(end_date))

        if full_data.empty or display_data.empty:
            st.error(f"Could not find data for **{selected_stock}**. Check the symbol and try again.")
            st.stop()
        if len(full_data) < 150:
            st.error("Not enough historical data. Try an earlier start date (at least 1–2 years of data needed).")
            st.stop()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        st.stop()

# ── Overview Metrics ──────────────────────────────────────────────────────────
latest  = float(display_data['Close'].iloc[-1])
prev    = float(display_data['Close'].iloc[-2])
change  = latest - prev
pct_chg = change / prev * 100
arrow   = "▲" if change >= 0 else "▼"
dcolor  = "normal" if change >= 0 else "inverse"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest Close", f"${latest:.2f}",
          f"{arrow} Daily Change: {change:+.2f} ({pct_chg:+.2f}%)", delta_color=dcolor)
c2.metric("Period High",  f"${float(display_data['High'].max()):.2f}")
c3.metric("Period Low",   f"${float(display_data['Low'].min()):.2f}")
c4.metric("Avg Volume",   f"{int(display_data['Volume'].mean()):,}")

# ── Candlestick + MAs + Volume ────────────────────────────────────────────────
ind_display = add_indicators(display_data.copy())

fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.75, 0.25], vertical_spacing=0.03)
fig_main.add_trace(go.Candlestick(
    x=display_data['Date'], open=display_data['Open'], high=display_data['High'],
    low=display_data['Low'], close=display_data['Close'],
    increasing_line_color='#00e676', decreasing_line_color='#ff5252', name='Price'
), row=1, col=1)

if len(ind_display) > 0:
    fig_main.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['MA_20'],
                                  line=dict(color='#ffab40', width=1.5), name='MA 20'), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['MA_50'],
                                  line=dict(color='#40c4ff', width=1.5), name='MA 50'), row=1, col=1)
    if ind_display['MA_200'].notna().any():
        fig_main.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['MA_200'],
                                      line=dict(color='#ea80fc', width=1.5), name='MA 200'), row=1, col=1)

vol_colors = ['#00e676' if c >= o else '#ff5252'
              for c, o in zip(display_data['Close'], display_data['Open'])]
fig_main.add_trace(go.Bar(x=display_data['Date'], y=display_data['Volume'],
                          marker_color=vol_colors, name='Volume', opacity=0.7), row=2, col=1)

fig_main.update_layout(
    title=f"{selected_stock} — Price, Moving Averages & Volume",
    template="plotly_dark", height=600,
    xaxis_rangeslider_visible=False,
    legend=dict(orientation='h', y=1.02, x=0),
    margin=dict(l=10, r=10, t=60, b=10)
)
fig_main.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig_main.update_yaxes(title_text="Volume",      row=2, col=1)
st.plotly_chart(fig_main, use_container_width=True)

# ── OHLCV Table ───────────────────────────────────────────────────────────────
st.subheader("OHLCV Data")
ohlcv = display_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
ohlcv['Date']         = pd.to_datetime(ohlcv['Date']).dt.strftime('%Y-%m-%d')
ohlcv['Daily Change'] = display_data['Close'].diff().values
ohlcv['Change %']     = display_data['Close'].pct_change().mul(100).values
for col in ['Open', 'High', 'Low', 'Close']:
    ohlcv[col] = ohlcv[col].apply(lambda x: f"${x:.2f}")
ohlcv['Volume']       = ohlcv['Volume'].apply(lambda x: f"{int(x):,}")
ohlcv['Daily Change'] = ohlcv['Daily Change'].apply(
    lambda x: f"▲ +${x:.2f}" if pd.notna(x) and x >= 0 else (f"▼ -${abs(x):.2f}" if pd.notna(x) else "—"))
ohlcv['Change %']     = ohlcv['Change %'].apply(
    lambda x: f"+{x:.2f}%" if pd.notna(x) and x >= 0 else (f"{x:.2f}%" if pd.notna(x) else "—"))
ohlcv = ohlcv.iloc[::-1].reset_index(drop=True)
st.dataframe(ohlcv, use_container_width=True, hide_index=True, height=300)

# ── Technical Indicators ──────────────────────────────────────────────────────
st.subheader("Technical Indicators")

tab_sma, tab_ema, tab_rsi, tab_macd, tab_bb, tab_stoch, tab_atr, tab_obv = st.tabs([
    "  SMA  ", "  EMA  ", "  RSI  ", "  MACD  ",
    "  Bollinger Bands  ", "  Stochastic  ", "  ATR  ", "  OBV  "
])

# ── SMA ───────────────────────────────────────────────────────────────────────
with tab_sma:
    fig_sma = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig_sma.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                 line=dict(color='#ffffff', width=2), name='Close'), row=1, col=1)
    fig_sma.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['SMA_20'],
                                 line=dict(color='#ffab40', width=1.5), name='SMA 20'), row=1, col=1)
    fig_sma.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['SMA_50'],
                                 line=dict(color='#40c4ff', width=1.5), name='SMA 50'), row=1, col=1)
    fig_sma.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['SMA_200'],
                                 line=dict(color='#ea80fc', width=1.5), name='SMA 200'), row=1, col=1)
    fig_sma.add_trace(go.Bar(x=ind_display['Date'], y=display_data['Volume'],
                             marker_color=vol_colors, opacity=0.7, name='Volume'), row=2, col=1)
    fig_sma.update_layout(template="plotly_dark", height=520,
                          title="Simple Moving Average (SMA 20 / 50 / 200)",
                          xaxis_rangeslider_visible=False,
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_sma.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_sma.update_yaxes(title_text="Volume",      row=2, col=1)
    st.plotly_chart(fig_sma, use_container_width=True)

# ── EMA ───────────────────────────────────────────────────────────────────────
with tab_ema:
    fig_ema = go.Figure()
    fig_ema.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                 line=dict(color='#ffffff', width=2), name='Close'))
    fig_ema.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['EMA_12'],
                                 line=dict(color='#00e676', width=1.5), name='EMA 12'))
    fig_ema.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['EMA_26'],
                                 line=dict(color='#ff6d00', width=1.5), name='EMA 26'))
    fig_ema.update_layout(template="plotly_dark", height=480,
                          title="Exponential Moving Average (EMA 12 / 26)",
                          yaxis_title="Price (USD)",
                          margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_ema, use_container_width=True)

# ── RSI ───────────────────────────────────────────────────────────────────────
with tab_rsi:
    fig_rsi = go.Figure()
    fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(255,82,82,0.08)',  line_width=0)
    fig_rsi.add_hrect(y0=0,  y1=30,  fillcolor='rgba(0,230,118,0.08)',  line_width=0)
    fig_rsi.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['RSI'],
                                 line=dict(color='#ffab40', width=2), name='RSI'))
    fig_rsi.add_hline(y=70, line_dash='dash', line_color='#ff5252',
                      annotation_text='Overbought (70)', annotation_position='top left')
    fig_rsi.add_hline(y=30, line_dash='dash', line_color='#00e676',
                      annotation_text='Oversold (30)', annotation_position='bottom left')
    fig_rsi.update_layout(template="plotly_dark", height=440, yaxis=dict(range=[0, 100]),
                          title="RSI (14-period)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_rsi, use_container_width=True)

# ── MACD ──────────────────────────────────────────────────────────────────────
with tab_macd:
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.5, 0.5], vertical_spacing=0.04)
    fig_macd.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                  line=dict(color='#40c4ff', width=2), name='Close'), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['MACD'],
                                  line=dict(color='#69f0ae', width=2), name='MACD'), row=2, col=1)
    fig_macd.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['MACD_Signal'],
                                  line=dict(color='#ff6d00', width=2), name='Signal'), row=2, col=1)
    bar_colors = ['#69f0ae' if v >= 0 else '#ff5252' for v in ind_display['MACD_Hist']]
    fig_macd.add_trace(go.Bar(x=ind_display['Date'], y=ind_display['MACD_Hist'],
                              marker_color=bar_colors, name='Histogram', opacity=0.8), row=2, col=1)
    fig_macd.update_layout(template="plotly_dark", height=520, title="MACD (12, 26, 9)",
                           margin=dict(l=10, r=10, t=50, b=10))
    fig_macd.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_macd.update_yaxes(title_text="MACD",        row=2, col=1)
    st.plotly_chart(fig_macd, use_container_width=True)

# ── Bollinger Bands ───────────────────────────────────────────────────────────
with tab_bb:
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['BB_Upper'],
                                line=dict(color='rgba(100,181,246,0.5)', dash='dash', width=1),
                                name='Upper Band'))
    fig_bb.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['BB_Lower'],
                                line=dict(color='rgba(100,181,246,0.5)', dash='dash', width=1),
                                fill='tonexty', fillcolor='rgba(100,181,246,0.07)',
                                name='Lower Band'))
    fig_bb.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['BB_Mid'],
                                line=dict(color='#64b5f6', width=1, dash='dot'), name='SMA 20'))
    fig_bb.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                line=dict(color='#ffffff', width=2), name='Close'))
    fig_bb.update_layout(template="plotly_dark", height=480,
                         title="Bollinger Bands (20-period, 2σ)",
                         margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_bb, use_container_width=True)

# ── Stochastic ────────────────────────────────────────────────────────────────
with tab_stoch:
    fig_stoch = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.55, 0.45], vertical_spacing=0.04)
    fig_stoch.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                   line=dict(color='#40c4ff', width=2), name='Close'), row=1, col=1)
    fig_stoch.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Stoch_K'],
                                   line=dict(color='#64b5f6', width=2), name='%K'), row=2, col=1)
    fig_stoch.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Stoch_D'],
                                   line=dict(color='#ff6d00', width=2), name='%D'), row=2, col=1)
    fig_stoch.add_hrect(y0=80, y1=100, fillcolor='rgba(255,82,82,0.08)',  line_width=0, row=2, col=1)
    fig_stoch.add_hrect(y0=0,  y1=20,  fillcolor='rgba(0,230,118,0.08)',  line_width=0, row=2, col=1)
    fig_stoch.add_hline(y=80, line_dash='dash', line_color='#ff5252',
                        annotation_text='Overbought (80)', annotation_position='top left',    row=2, col=1)
    fig_stoch.add_hline(y=20, line_dash='dash', line_color='#00e676',
                        annotation_text='Oversold (20)',   annotation_position='bottom left', row=2, col=1)
    fig_stoch.update_layout(template="plotly_dark", height=520,
                            title="Stochastic Oscillator (%K / %D)",
                            margin=dict(l=10, r=10, t=50, b=10))
    fig_stoch.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_stoch.update_yaxes(title_text="Stochastic", range=[0, 100], row=2, col=1)
    st.plotly_chart(fig_stoch, use_container_width=True)

# ── ATR ───────────────────────────────────────────────────────────────────────
with tab_atr:
    fig_atr = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.04)
    fig_atr.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                 line=dict(color='#40c4ff', width=2), name='Close'), row=1, col=1)
    fig_atr.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['ATR'],
                                 line=dict(color='#ffab40', width=2), name='ATR (14)',
                                 fill='tozeroy', fillcolor='rgba(255,171,64,0.15)'), row=2, col=1)
    fig_atr.update_layout(template="plotly_dark", height=500,
                          title="Average True Range / Volatility (ATR 14)",
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_atr.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_atr.update_yaxes(title_text="ATR ($)",     row=2, col=1)
    st.plotly_chart(fig_atr, use_container_width=True)

# ── OBV ───────────────────────────────────────────────────────────────────────
with tab_obv:
    fig_obv = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.03)
    fig_obv.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['Close'],
                                 line=dict(color='#40c4ff', width=2), name='Close'), row=1, col=1)
    fig_obv.add_trace(go.Bar(x=ind_display['Date'], y=display_data['Volume'],
                             marker_color=vol_colors, opacity=0.75, name='Volume'), row=2, col=1)
    fig_obv.add_trace(go.Scatter(x=ind_display['Date'], y=ind_display['OBV'],
                                 line=dict(color='#ea80fc', width=2), name='OBV'), row=3, col=1)
    fig_obv.update_layout(template="plotly_dark", height=580,
                          title="On-Balance Volume (OBV)",
                          margin=dict(l=10, r=10, t=50, b=10))
    fig_obv.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig_obv.update_yaxes(title_text="Volume",      row=2, col=1)
    fig_obv.update_yaxes(title_text="OBV",         row=3, col=1)
    st.plotly_chart(fig_obv, use_container_width=True)

# ── Train & Predict ───────────────────────────────────────────────────────────
if predict_btn:

    if cache_key in st.session_state.model_cache:
        # ── Load from cache ───────────────────────────────────────────────────
        st.info("✅ Using cached model — skipping training.")
        lstm_model, scaler, full_dates_cache = st.session_state.model_cache[cache_key]

        close_vals   = full_data['Close'].values.reshape(-1, 1)
        close_scaled = scaler.transform(close_vals)

    else:
        # ── Train fresh model for this stock ──────────────────────────────────
        st.markdown("### 🔧 Training Model")
        st.caption(f"Training on **{selected_stock}** data from {start_date} → today. This takes ~2–5 minutes.")

        close_vals   = full_data['Close'].values.reshape(-1, 1)
        scaler       = MinMaxScaler((0, 1))
        close_scaled = scaler.fit_transform(close_vals)

        X_all, y_all = build_sequences(close_scaled, LOOKBACK)

        split_idx = int(len(X_all) * 0.8)
        X_train   = X_all[:split_idx]
        y_train   = y_all[:split_idx]

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        lstm_model = build_model(input_shape=(X_train.shape[1], 1))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, verbose=0),
        ]

        progress_bar = st.progress(0, text="Training…")

        class ProgressCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                pct = int((epoch + 1) / self.params['epochs'] * 100)
                progress_bar.progress(pct,
                    text=f"Epoch {epoch+1}/{self.params['epochs']}  ·  "
                         f"loss: {logs.get('loss', 0):.5f}  ·  "
                         f"val_loss: {logs.get('val_loss', 0):.5f}")

        history = lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=10,
            validation_split=0.15,
            callbacks=callbacks + [ProgressCB()],
            verbose=0
        )
        progress_bar.empty()

        full_dates_cache = pd.to_datetime(full_data['Date'])
        st.session_state.model_cache[cache_key] = (lstm_model, scaler, full_dates_cache)

        # Loss chart
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=history.history['loss'],
                                      line=dict(color='#40c4ff', width=2), name='Train Loss'))
        fig_loss.add_trace(go.Scatter(y=history.history['val_loss'],
                                      line=dict(color='#ff6d00', width=2), name='Val Loss'))
        fig_loss.update_layout(title="Training & Validation Loss",
                               xaxis_title="Epoch", yaxis_title="MSE Loss",
                               template="plotly_dark", height=400,
                               margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── Run predictions for display window ────────────────────────────────────
    full_dates = pd.to_datetime(full_data['Date'])
    disp_start = pd.to_datetime(start_date)
    disp_end   = pd.to_datetime(end_date)

    disp_mask    = (full_dates >= disp_start) & (full_dates <= disp_end)
    disp_indices = full_data.index[disp_mask].tolist()

    if len(disp_indices) == 0:
        st.error("No data in selected display window.")
        st.stop()

    # Silently skip indices that don't have enough lookback history
    disp_indices = [i for i in disp_indices if i >= LOOKBACK]

    if len(disp_indices) == 0:
        st.error("Start date is too close to the beginning of the data — "
                 "the first 60 trading days are needed as lookback. "
                 "Try moving your start date forward by ~3 months.")
        st.stop()

    X_disp, y_disp = [], []
    for i in disp_indices:
        X_disp.append(close_scaled[i - LOOKBACK:i, 0])
        y_disp.append(close_scaled[i, 0])

    X_disp = np.array(X_disp).reshape(-1, LOOKBACK, 1)
    y_disp = np.array(y_disp)

    with st.spinner("Generating predictions..."):
        y_pred_scaled    = lstm_model.predict(X_disp, verbose=0)
        predicted_prices = scaler.inverse_transform(y_pred_scaled)
        actual_prices    = scaler.inverse_transform(y_disp.reshape(-1, 1))

    mse  = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actual_prices, predicted_prices)
    r2   = r2_score(actual_prices, predicted_prices)

    st.markdown("### 📉 Predictions vs Actual")
    plot_dates = full_dates[disp_indices]

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=plot_dates, y=actual_prices.flatten(),
        mode='lines', name='Actual',
        line=dict(color='#ff5252', width=2.5)
    ))
    fig_pred.add_trace(go.Scatter(
        x=plot_dates, y=predicted_prices.flatten(),
        mode='lines', name='Predicted',
        line=dict(color='#69f0ae', width=2.5)
    ))
    fig_pred.update_layout(
        title=f"{selected_stock}: Actual vs Predicted",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=560,
        legend=dict(orientation='h', y=1.05),
        margin=dict(l=10, r=10, t=60, b=10)
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # Metrics table under graph
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MSE", "R²"],
        "Value":  [f"${rmse:.2f}", f"${mae:.2f}", f"{mse:.2f}", f"{r2:.4f}"]
    })
    st.dataframe(metrics_df, use_container_width=False, hide_index=True, width=300)