import sys, subprocess, importlib, os

__author__ = "Prophacy AI"


# --------- bootstrap (existing) ----------
def ensure_package(pkg):
    try:
        importlib.import_module(pkg)
        return False
    except Exception:
        print(f"[!] {pkg} not found, installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        except subprocess.CalledProcessError as e:
            print(f"[!] pip install failed for {pkg}: {e}")
            sys.exit(1)
        return True


# Fix: Changed "beautifulsoup4" to "bs4" to prevent endless loop
# Add FinBERT dependencies
packages = ["yfinance", "plotly", "pandas", "prophet", "cmdstanpy", "streamlit", "numpy", "textblob", "requests", "bs4",
            "transformers", "torch", "dotenv", "alpaca"]
restart_needed = any(ensure_package(pkg) for pkg in packages)

if restart_needed:
    print("[!] New packages installed. Please restart the script.")
    sys.exit(0)

# ---------- Imports ----------
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
from prophet import Prophet
from datetime import date, timedelta
import warnings
import torch

# Agent imports
from agent.agent import load_state, save_state, maybe_auto_run, portfolio_value
from agent.strategy import get_targets
import datetime as dt

# Alpaca API imports
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

warnings.filterwarnings('ignore')


# ---------- Helper Functions ----------
def prepare_price_df(raw_df: pd.DataFrame, ticker: str):
    """
    Make a clean 2-col dataframe: ds (datetime) and y (price).
    Handles yfinance MultiIndex or missing 'Adj Close'.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), None

    df = raw_df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join([str(x) for x in tup if x is not None]) for tup in df.columns.to_list()]

    cols = list(df.columns)

    # Prefer adjusted close, fallback to close
    candidates = [c for c in cols if "Adj Close" in c] or [c for c in cols if "Close" in c]
    if ticker and candidates:
        candidates = sorted(candidates, key=lambda c: (ticker not in c, len(c)))

    if not candidates:
        raise ValueError("No price column found.")

    price_col = candidates[0]

    out = (
        df.reset_index()
            .rename(columns={"Date": "ds"})[["ds", price_col]]
            .rename(columns={price_col: "y"})
    )
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out.dropna().sort_values("ds"), price_col


# ---------- Alpaca API Connection ----------
def init_alpaca_api():
    """Initialize and test Alpaca API connection"""
    try:
        # Load environment variables from .env file
        print("Loading .env file...")
        load_dotenv()
        print("Environment loaded. Checking variables...")
        
        # Debug: Check all environment variables first
        print("All APCA environment variables:")
        for key, value in os.environ.items():
            if 'APCA' in key:
                print(f"  {key}: {repr(value)}")
        
        # Grab credentials
        BASE_URL = os.getenv("APCA_API_BASE_URL")
        API_KEY = os.getenv("APCA_API_KEY_ID")
        API_SECRET = os.getenv("APCA_API_SECRET_KEY")
        
        # Debug: Print environment variables
        print("After os.getenv():")
        print("BASE_URL:", repr(BASE_URL))
        print("API_KEY:", repr(API_KEY))
        print("API_SECRET:", repr(API_SECRET))
        
        # Initialize Alpaca Trading Client
        trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        
        # Initialize Alpaca Data Client
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        
        # Quick test
        account = trading_client.get_account()
        print("✅ Connected to Alpaca")
        print("Account ID:", account.id)
        print("Cash Balance:", account.cash)
        
        return trading_client, data_client
    except Exception as e:
        print("❌ Alpaca connection failed:", e)
        return None, None

# Initialize Alpaca API
trading_client, data_client = init_alpaca_api()


# ---------- FinBERT Setup ----------
@st.cache_resource
def load_finbert():
    """Load FinBERT model once and cache it"""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load FinBERT: {e}")
        return None, None


# Load FinBERT
finbert_tokenizer, finbert_model = load_finbert()


def finbert_sentiment(text):
    """Analyze sentiment using FinBERT"""
    if finbert_tokenizer is None or finbert_model is None:
        return "neutral", 0.0

    try:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = finbert_model(**inputs)

        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        labels = ['negative', 'neutral', 'positive']
        sentiment = labels[torch.argmax(scores)]
        score = scores[0][torch.argmax(scores)].item()

        # Convert to polarity scale
        if sentiment == "positive":
            polarity = score
        elif sentiment == "negative":
            polarity = -score
        else:
            polarity = 0

        return sentiment, polarity
    except Exception as e:
        return "neutral", 0.0


# ---------- Enhanced Sentiment Analysis Functions ----------
def aggregate_daily_sentiment(sentiment_df):
    """Enhanced daily sentiment aggregation that preserves strong signals"""
    if sentiment_df.empty:
        return sentiment_df

    daily_sentiment = sentiment_df.groupby('date').agg({
        'sentiment': lambda x: np.sign(x).mean() + 0.5 * (np.sign(x).sum() / len(x)),
        'title': lambda x: " | ".join(x[:3]),
        'source': 'first'
    }).reset_index()

    return daily_sentiment


def get_news_sentiment(ticker, start_date, end_date):
    """
    Fetch news sentiment for the given ticker and date range with fallback sources
    """
    # First try: Yahoo Finance
    yahoo_sentiment = _get_yahoo_sentiment(ticker, start_date, end_date)

    # If Yahoo fails, try NewsAPI as fallback
    if yahoo_sentiment.empty:
        yahoo_sentiment = _get_newsapi_sentiment(ticker, start_date, end_date)

    # If both fail, create synthetic neutral sentiment
    if yahoo_sentiment.empty:
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        yahoo_sentiment = pd.DataFrame({
            'date': date_range,
            'sentiment': [0.0] * len(date_range),
            'title': ['No news available'] * len(date_range),
            'source': ['synthetic'] * len(date_range)
        })
        st.info("Using synthetic neutral sentiment due to limited news data.")

    # Apply enhanced aggregation
    return aggregate_daily_sentiment(yahoo_sentiment)


def _get_yahoo_sentiment(ticker, start_date, end_date):
    """Fetch sentiment from Yahoo Finance JSON API"""
    try:
        import requests
        import pandas as pd
        from datetime import datetime
        # assumes you already have finbert_sentiment(title)

        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={ticker}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = []

            for item in data.get("news", [])[:20]:  # limit to 20 articles
                title = item.get("title")
                link = item.get("link")
                pubtime = item.get("providerPublishTime")  # Unix timestamp

                if title:
                    sentiment, polarity = finbert_sentiment(title)
                    articles.append({
                        "date": datetime.fromtimestamp(pubtime).date()
                        if pubtime else datetime.today().date(),
                        "sentiment": polarity,
                        "title": title,
                        "source": item.get("publisher", "Yahoo Finance"),
                        "link": link
                    })

            return pd.DataFrame(articles)
        else:
            st.warning(f"Yahoo Finance API error {response.status_code}")
    except Exception as e:
        st.warning(f"Yahoo Finance news fetch failed: {e}")

    return pd.DataFrame()


def _get_newsapi_sentiment(ticker, start_date, end_date):
    """Fallback sentiment from NewsAPI (requires API key)"""
    # This would require a NewsAPI key
    # For now, return empty DataFrame
    return pd.DataFrame()


def create_sentiment_regressor(sentiment_df, forecast_dates):
    """Create sentiment regressor for Prophet forecasting with guaranteed coverage"""
    if sentiment_df.empty:
        return pd.DataFrame({
            'ds': forecast_dates,
            'sentiment': [0.0] * len(forecast_dates)
        })

    # Interpolate sentiment for forecast dates
    all_dates = pd.concat([
        sentiment_df[['date', 'sentiment']].rename(columns={'date': 'ds'}),
        pd.DataFrame({'ds': forecast_dates, 'sentiment': [0.0] * len(forecast_dates)})
    ]).drop_duplicates(subset=['ds']).sort_values('ds')

    # Forward fill sentiment values, then fill remaining gaps with 0
    all_dates['sentiment'] = all_dates['sentiment'].fillna(method='ffill').fillna(0)

    # Filter to only forecast dates
    future_sentiment = all_dates[all_dates['ds'].isin(forecast_dates)]

    return future_sentiment


def build_features(price_df, sentiment_df):
    """Build enhanced features for Prophet including sentiment lags and news volume"""
    if sentiment_df.empty:
        # Create neutral sentiment if none available
        sentiment_df = pd.DataFrame({
            'date': price_df['ds'],
            'sentiment': [0.0] * len(price_df)
        })
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # Also (if needed) normalize to date only (no time part)
    sentiment_df['date'] = sentiment_df['date'].dt.normalize()

    # Ensure price_df.ds is also datetime64[ns]
    price_df['ds'] = pd.to_datetime(price_df['ds'])
    price_df['ds'] = price_df['ds'].dt.normalize()
    # Merge sentiment with prices
    merged = pd.merge(price_df, sentiment_df, left_on='ds', right_on='date', how='left')
    merged['sentiment'] = merged['sentiment'].fillna(0)

    # Create lagged features
    merged['sentiment_lag1'] = merged['sentiment'].shift(1).fillna(0)
    merged['sentiment_lag3'] = merged['sentiment'].shift(3).fillna(0)

    # News volume (count of news items per day)
    if not sentiment_df.empty:
        news_volume = sentiment_df.groupby('date').size().reset_index(name='news_volume')
        merged = pd.merge(merged, news_volume, left_on='ds', right_on='date', how='left')
        merged['news_volume'] = merged['news_volume'].fillna(0)
    else:
        merged['news_volume'] = 0

    # Returns and volatility
    merged['ret1'] = merged['y'].pct_change(1).fillna(0)
    merged['ret5'] = merged['y'].pct_change(5).fillna(0)
    merged['vol20'] = merged['y'].rolling(20).std().fillna(0)

    # Event flags
    merged['covid_crash'] = ((merged['ds'] >= "2020-02-20") & (merged['ds'] <= "2020-04-01")).astype(int)

    return merged


# ---------- Streamlit App ----------
st.set_page_config(
    page_title="Prophacy AI - Stock Prediction with FinBERT Sentiment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark, professional theme ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0b0f18;            /* app background (neutral, not blue-heavy) */
        --panel: #11151f;         /* cards/panels */
        --panel-2: #141a24;       /* inputs */
        --text: #e5e9f0;          /* primary text */
        --muted: #a0a8b5;         /* secondary text */
        --stroke: #222838;        /* borders */
        --steel: #9fb3c8;         /* soft steel for historical lines */
        --accent-teal: #56D4AA;   /* color pop 1 */
        --accent-purple: #B388FF; /* color pop 2 */
        --accent-amber: #FFB81C;  /* matches confidence band */
        --btn: #2979FF;           /* main clickable (graphite) */
        --btn-hover: #1565C0;     /* main clickable hover */
    }

    /* App background */
    [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }

    /* Header */
    [data-testid="stHeader"] { background: linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0)); }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0a0e16; border-right: 1px solid var(--stroke); }

    /* Headings */
    h1, h2, h3, h4 { color: var(--text); }

    /* Accent divider */
    .accent-bar { height: 3px; border-radius: 999px; background: linear-gradient(90deg, var(--accent-teal), var(--accent-purple)); margin-top: 10px; opacity: 0.9; }

    /* Cards/metrics/alerts */
    .stMetric, .stAlert, .stInfo, .stSuccess, .stWarning, .stError {
        background: var(--panel);
        border: 1px solid var(--stroke);
        color: var(--text);
        border-radius: 14px;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 6px 18px rgba(0,0,0,0.35);
    }

    /* Subtle colored edge on info cards to add life */
    .stAlert { border-left: 2px solid var(--accent-teal); }

    /* Metric numbers: keep smaller; add slight pop */
    [data-testid="stMetricValue"] { font-size: 1.6rem; line-height: 1.1; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem; color: var(--muted); }
    [data-testid="stMetric"] { padding: 12px 16px; margin-bottom: 16px; border-radius: 14px; border: 1px solid var(--stroke); }

    /* Tabs: more spacing + colorful active indicator */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; padding: 10px 4px; border-bottom: 1px solid var(--stroke); }
    .stTabs [data-baseweb="tab"] { background: var(--panel); color: var(--muted); border-radius: 12px 12px 0 0; padding: 12px 18px; border: 1px solid var(--stroke); border-bottom: none; }
    .stTabs [aria-selected="true"] {
        background: var(--panel-2);
        color: var(--text);
        box-shadow: inset 0 -3px 0 0 var(--accent-amber);
        border-image: linear-gradient(90deg, var(--accent-teal), var(--accent-purple)) 1;
        border-top: 1px solid transparent; /* unify with border-image */
    }

    /* Dataframes */
    .stDataFrame, .dataframe { background: var(--panel); color: var(--text); border: 1px solid var(--stroke); border-radius: 12px; }

    /* Inputs */
    input, textarea, select { background: var(--panel-2) !important; color: var(--text) !important; border: 1px solid var(--stroke) !important; border-radius: 10px !important; }
    .stTextInput>div>div>input, .stDateInput>div>div>input { height: 42px; padding: 8px 12px; }

    /* Sliders */
    [data-baseweb="slider"] [aria-hidden="true"] { background: var(--panel-2); }
    [data-baseweb="slider"] [role="slider"] { outline: none; box-shadow: 0 0 0 2px rgba(86,212,170,.25); }

    /* Buttons - MAIN content: add accent ring on hover */
    [data-testid="stAppViewContainer"] .stButton>button,
    [data-testid="stAppViewContainer"] .stDownloadButton>button {
        background: var(--btn);
        color: white;
        border: 1px solid #242a36;
        border-radius: 12px;
        font-weight: 600;
        transition: box-shadow .15s ease, background .15s ease, transform .05s ease;
    }
    [data-testid="stAppViewContainer"] .stButton>button:hover,
    [data-testid="stAppViewContainer"] .stDownloadButton>button:hover {
        background: var(--btn-hover);
        box-shadow: 0 0 0 2px rgba(86,212,170,.25), 0 8px 20px rgba(0,0,0,.35);
    }

    /* Buttons - SIDEBAR: slightly dimmer */
    [data-testid="stSidebar"] .stButton>button,
    [data-testid="stSidebar"] .stDownloadButton>button { background: #1e2430; color: var(--text); border: 1px solid var(--stroke); border-radius: 12px; }
    [data-testid="stSidebar"] .stButton>button:hover,
    [data-testid="stSidebar"] .stDownloadButton>button:hover { background: #272f3e; box-shadow: 0 0 0 2px rgba(179,136,255,.25); }

    /* Section header pill with accented border */
    .section-header { background: linear-gradient(135deg, #0e121b 0%, #141a24 100%); color: var(--text); padding: 14px 22px; border-radius: 14px; margin: 18px 0; font-weight: 800; text-align: center; border: 1px solid var(--stroke); box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 6px 18px rgba(0,0,0,0.35); border-top: 2px solid var(--accent-teal); }

    /* Footer */
    .footer { color: var(--muted); background: rgba(255,255,255,0.02); border: 1px solid var(--stroke); border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header (no emojis)
st.markdown(
    """
    <div style="background:linear-gradient(135deg, #0e121b 0%, #141a24 100%); padding: 30px; border-radius: 15px; text-align: center; margin-bottom: 18px; box-shadow: 0 8px 32px rgba(0,0,0,0.25); border: 1px solid #222838;">
        <h1 style="color: #E5E9F0; margin: 0; font-size: 2.6em; font-weight: 900;">Prophacy AI</h1>
        <p style="color: #A0A8B5; font-size: 1.1em; margin: 0; font-weight: 400;">Stock Prediction with FinBERT Sentiment Analysis</p>
        <div class="accent-bar"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar Configuration ----------
with st.sidebar:
    st.markdown(
        """
        <div style="background:linear-gradient(135deg, #0e121b 0%, #0b0f18 100%); padding: 15px; border-radius: 15px; text-align: center; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.25); border: 1px solid #222838;">
            <h2 style="color: #E5E9F0; margin: 0;">Configuration</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Ticker Selection with dropdown
    st.markdown("### Ticker Selection")
    
    # Top 5 companies plus custom option
    top_companies = {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Google (GOOGL)": "GOOGL",
        "Tesla (TSLA)": "TSLA",
        "Custom...": "CUSTOM"
    }
    
    selected_company = st.selectbox(
        "Select Company",
        list(top_companies.keys()),
        index=0,
        help="Select from top companies or choose custom to enter your own ticker"
    )
    
    if top_companies[selected_company] == "CUSTOM":
        ticker = st.text_input("Enter Custom Ticker", value="AAPL", placeholder="AAPL")
    else:
        ticker = top_companies[selected_company]

    # Prediction Settings
    st.markdown("### Prediction Settings")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=date.today())

    forecast_days = st.slider("Forecast Period (Days)", min_value=30, max_value=365, value=90, step=30,
                              help="Number of days to forecast into the future")

    # Sentiment Analysis
    st.markdown("### Sentiment Analysis")
    enable_sentiment = st.checkbox("Enable FinBERT Sentiment Analysis", value=True,
                                   help="Use news sentiment to improve predictions")

    # Mode Selection
    st.markdown("### Mode Selection")
    mode = st.radio(
        "Choose Analysis Mode",
        ["Long-term Forecast", "Showstopper (Next-Day Demo)", "Autopilot Investing (Agent)"],
        index=0,
        help="Long-term: Multi-day forecast | Showstopper: Next-day prediction demo | Autopilot: fully automated investing"
    )

    # Info Section
    st.markdown("---")
    st.markdown("### About")
    st.info(
        """
        Prophacy AI combines:
        - Prophet forecasting
        - FinBERT sentiment analysis
        - Real-time market data
        """
    )

# Main Content Area
if mode == "Long-term Forecast":
    with st.container():
        st.markdown("<div class=\"section-header\">Long-term Stock Forecast</div>", unsafe_allow_html=True)

        # Info cards at the top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("Selected Stock: " + ticker.upper())
        with col2:
            st.info(f"Period: {start_date} to {end_date}")
        with col3:
            st.info(f"Forecast: {forecast_days} days")

        with st.container():
            st.markdown("**Generate Forecast**")
            if st.button("Generate Prediction", type="primary", use_container_width=True):
                with st.spinner("Generating forecast..."):
                    # Create progress bar
                    progress_bar = st.progress(0)

                    # Download stock data
                    progress_bar.progress(20)
                    
                    # Input validation
                    ticker = ticker.strip()  # Strip any spaces from ticker
                    if start_date >= end_date:
                        st.error("Start date must be before end date. Please adjust your date selection.")
                        st.stop()
                    
                    data = yf.download(ticker, start=start_date, end=end_date)

                    if data.empty:
                        st.error("No stock data found, please check ticker and dates.")
                    else:
                        # Prepare data using helper function
                        progress_bar.progress(40)
                        df, price_col = prepare_price_df(data, ticker)

                        if df.empty:
                            st.error("Failed to prepare price data.")
                        else:
                            # Get sentiment data
                            progress_bar.progress(60)
                            sentiment_df = pd.DataFrame()
                            if enable_sentiment:
                                sentiment_df = get_news_sentiment(ticker, start_date, end_date)

                            # Build enhanced features
                            progress_bar.progress(80)
                            price_plus = build_features(df[['ds', 'y']], sentiment_df)

                            # Train Prophet model
                            if enable_sentiment:
                                model = Prophet(daily_seasonality=True)
                                model.add_regressor('sentiment')
                                model.add_regressor('sentiment_lag1')
                                model.add_regressor('news_volume')

                                prophet_train_cols = ['ds', 'y', 'sentiment', 'sentiment_lag1', 'news_volume']
                                model.fit(price_plus[prophet_train_cols])
                            else:
                                model = Prophet(daily_seasonality=True)
                                model.fit(df[['ds', 'y']])

                            # Create future dataframe
                            future = model.make_future_dataframe(periods=forecast_days, freq="B")

                            if enable_sentiment:
                                future_feats = build_features(
                                    price_plus[['ds', 'y']],
                                    sentiment_df
                                )[['ds', 'sentiment', 'sentiment_lag1', 'news_volume']]

                                future = future.merge(future_feats, on='ds', how='left').fillna(0.0)

                            # Generate forecast
                            forecast = model.predict(future)
                            progress_bar.progress(100)

                            # Display results in tabs
                            tab1, tab2, tab3, tab4 = st.tabs(["Forecast Chart", "Data", "Sentiment Analysis", "Forecast Details"])

                    with tab1:
                        st.subheader("Stock Price Forecast")
                        fig = go.Figure()

                        # Historical data - steel blue
                        fig.add_trace(go.Scatter(
                            x=df['ds'],
                            y=df['y'],
                            mode='lines',
                            name='Historical Price',
                            line=dict(color='#4682B4', width=3)  # Steel blue
                        ))

                        # Past prediction (model fit on historical data) - orange
                        # Get the forecast for the historical period
                        historical_forecast = forecast[forecast['ds'] <= df['ds'].max()]
                        fig.add_trace(go.Scatter(
                            x=historical_forecast['ds'],
                            y=historical_forecast['yhat'],
                            mode='lines',
                            name='Past Prediction',
                            line=dict(color='#FF8C00', width=3)  # Dark orange
                        ))

                        # Future prediction - green
                        future_forecast = forecast[forecast['ds'] > df['ds'].max()]
                        if not future_forecast.empty:
                            # Confidence band (light green)
                            fig.add_trace(go.Scatter(
                                x=future_forecast['ds'],
                                y=future_forecast['yhat_lower'],
                                mode='lines',
                                line=dict(color='#90EE90', width=1),
                                name='Confidence Lower',
                                showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=future_forecast['ds'],
                                y=future_forecast['yhat_upper'],
                                mode='lines',
                                line=dict(color='#90EE90', width=1),
                                name='Confidence Upper',
                                fill='tonexty',
                                fillcolor='rgba(144, 238, 144, 0.3)',
                                showlegend=False
                            ))

                            # Predicted line (dark green)
                            fig.add_trace(go.Scatter(
                                x=future_forecast['ds'],
                                y=future_forecast['yhat'],
                                mode='lines',
                                name='Future Forecast',
                                line=dict(color='#228B22', width=4)  # Forest green
                            ))

                        # Layout: dark background + subtle grid, colored axis lines
                        fig.update_layout(
                            title=f"{ticker.upper()} Stock Price Forecast",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            hovermode='x unified',
                            height=600,
                            plot_bgcolor='#0b0f18',
                            paper_bgcolor='#0b0f18',
                            font=dict(color='#E5E9F0'),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            xaxis=dict(gridcolor='#2a3142', zerolinecolor='#2a3142', showspikes=True, linecolor='#56D4AA'),
                            yaxis=dict(gridcolor='#2a3142', zerolinecolor='#2a3142', linecolor='#B388FF')
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.subheader("Historical Data Preview")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Data Points", len(df))
                            st.metric("Start Date", df['ds'].min().strftime('%Y-%m-%d'))
                        with col2:
                            st.metric("Current Price", f"${df['y'].iloc[-1]:.2f}")
                            st.metric("End Date", df['ds'].max().strftime('%Y-%m-%d'))

                        st.dataframe(df.tail(10), use_container_width=True)

                    with tab3:
                        if enable_sentiment and not sentiment_df.empty:
                            st.subheader("FinBERT Sentiment Analysis")

                            # Calculate metrics
                            avg_sentiment = sentiment_df['sentiment'].mean()
                            sentiment_std = sentiment_df['sentiment'].std()
                            positive_count = len(sentiment_df[sentiment_df['sentiment'] > 0])
                            negative_count = len(sentiment_df[sentiment_df['sentiment'] < 0])

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                            with col2:
                                st.metric("Sentiment Volatility", f"{sentiment_std:.3f}")
                            with col3:
                                st.metric("Positive News", positive_count)
                            with col4:
                                st.metric("Negative News", negative_count)

                            # Sentiment timeline - use teal for a bit more color
                            st.subheader("Sentiment Over Time")
                            fig_sent = px.line(
                                sentiment_df, x='date', y='sentiment',
                                title='Daily Sentiment Score', labels={'sentiment': 'Sentiment Score', 'date': 'Date'}
                            )
                            fig_sent.update_traces(line=dict(color='#56D4AA', width=3))
                            fig_sent.update_layout(
                                plot_bgcolor='#0b0f18', paper_bgcolor='#0b0f18', font=dict(color='#E5E9F0'),
                                xaxis=dict(gridcolor='#2a3142', linecolor='#56D4AA'),
                                yaxis=dict(gridcolor='#2a3142', linecolor='#B388FF')
                            )
                            st.plotly_chart(fig_sent, use_container_width=True)

                            # Recent sentiment with color coding
                            st.subheader("Recent News Headlines")
                            for _, row in sentiment_df.tail(5).iterrows():
                                sentiment = row["sentiment"]
                                if sentiment > 0.1:
                                    color = "#56D4AA"  # Teal for positive
                                    sentiment_label = "Positive"
                                elif sentiment < -0.1:
                                    color = "#FF6B6B"  # Red for negative
                                    sentiment_label = "Negative"
                                else:
                                    color = "#A0A8B5"  # Muted for neutral
                                    sentiment_label = "Neutral"
                                
                                st.markdown(
                                    f"""
                                    <div style="background-color: rgba{(*[int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)], 0.1)}; 
                                                padding: 10px; 
                                                border-radius: 8px; 
                                                border-left: 4px solid {color};
                                                margin-bottom: 10px;">
                                        <div style="color: {color}; font-weight: bold; margin-bottom: 5px;">
                                            {row['date'].strftime('%Y-%m-%d')}: {sentiment_label} (Score: {sentiment:.2f})
                                        </div>
                                        <div style="color: #E5E9F0;">
                                            {row['title'][:100]}...
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                        else:
                            st.info("Sentiment analysis is disabled or no sentiment data available.")

                    with tab4:
                        st.subheader("Forecast Details")

                        # Forecast metrics
                        last_forecast = forecast.iloc[-1]
                        current_price = df['y'].iloc[-1]
                        forecast_price = last_forecast['yhat']
                        change_percent = ((forecast_price - current_price) / current_price) * 100

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Current Price", f"${current_price:.2f}")
                        with col2:
                            st.metric("Forecast Price", f"${forecast_price:.2f}", f"{change_percent:+.2f}%")
                        with col3:
                            st.metric("Forecast Date", last_forecast['ds'].strftime('%Y-%m-%d'))

                        # Download forecast
                        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
                        st.download_button(
                            label="Download Forecast CSV",
                            data=csv,
                            file_name=f"{ticker}_forecast.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        # Show forecast table
                        st.dataframe(
                            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10), use_container_width=True
                        )

# ---------- Showstopper Mode ----------
elif mode == "Showstopper (Next-Day Demo)":
    with st.container():
        st.markdown("<div class=\"section-header\">Next-Day Prediction Demo</div>", unsafe_allow_html=True)

        st.info("This mode demonstrates the model's ability to predict the next day's stock price. It trains on data up to yesterday and predicts today's price.")

    if ticker:
        cutoff_date = date.today() - timedelta(days=1)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Selected Stock", ticker.upper())
        with col2:
            st.metric("Cutoff Date", cutoff_date.strftime('%Y-%m-%d'))

        with st.container():
            st.markdown("**Generate Next-Day Prediction**")
            if st.button("Predict Tomorrow's Price", type="primary", use_container_width=True):
                with st.spinner("Training model and generating prediction..."):
                    data_show = yf.download(ticker, start=start_date, end=cutoff_date)

                    if data_show.empty:
                        st.warning("Not enough data for showstopper mode.")
                    else:
                        # Prepare data using helper function
                        df_show, price_col = prepare_price_df(data_show, ticker)

                        if df_show.empty:
                            st.error("Failed to prepare price data for showstopper mode.")
                        else:
                            # Train Prophet on all data up to yesterday
                            model_show = Prophet(daily_seasonality=True)
                            model_show.fit(df_show)

                            # Forecast just 1 day ahead (today)
                            future_show = model_show.make_future_dataframe(periods=1, freq="B")
                            forecast_show = model_show.predict(future_show)

                            # Save forecast for use in the hackathon demo
                            forecast_file = f"forecast_{ticker}_{cutoff_date}.csv"
                            forecast_show[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(forecast_file, index=False)

                            # Display prediction
                            last_price = df_show['y'].iloc[-1]
                            predicted_price = forecast_show['yhat'].iloc[-1]
                            change = predicted_price - last_price
                            change_percent = (change / last_price) * 100

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Last Price", f"${last_price:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${predicted_price:.2f}", f"{change_percent:+.2f}%")
                            with col3:
                                st.metric("Prediction Date", forecast_show['ds'].iloc[-1].strftime('%Y-%m-%d'))

                            # Try to compare with today's actual price if available
                            actual_today = yf.download(ticker, start=cutoff_date, end=date.today())
                            if not actual_today.empty:
                                # Use helper function for actual data
                                actual_df, _ = prepare_price_df(actual_today, ticker)

                                if not actual_df.empty:
                                    # Convert ds to datetime with tz removed before merging
                                    forecast_show["ds"] = pd.to_datetime(forecast_show["ds"]).dt.tz_localize(None)
                                    actual_df["ds"] = pd.to_datetime(actual_df["ds"]).dt.tz_localize(None)

                                    # Safe merge on ds
                                    merged = pd.merge(
                                        forecast_show[["ds", "yhat"]], actual_df[["ds", "y"]], on="ds", how="inner"
                                    )

                                    if not merged.empty:
                                        st.subheader("Prediction vs Actual")
                                        actual_price = merged['y'].iloc[0]
                                        accuracy = 100 - (abs(actual_price - predicted_price) / actual_price * 100)

                                        fig_comp = go.Figure()
                                        fig_comp.add_trace(go.Indicator(
                                            mode="number+delta",
                                            value=predicted_price,
                                            number={'prefix': "$"},
                                            delta={'reference': actual_price, 'relative': True, 'valueformat': '.2%'},
                                            title={"text": "Prediction Accuracy"}
                                        ))
                                        fig_comp.update_layout(
                                            height=300,
                                            plot_bgcolor='#0b0f18',
                                            paper_bgcolor='#0b0f18',
                                            font=dict(color='#E5E9F0')
                                        )
                                        st.plotly_chart(fig_comp, use_container_width=True)

                                        st.metric("Model Accuracy", f"{accuracy:.1f}%")
                                    else:
                                        st.info("Today's actual data not fully available yet. Try again later.")
                                else:
                                    st.info("Failed to prepare actual price data for comparison.")
                            else:
                                st.info("Today's actual data not available yet. Try again later.")
    else:
        st.info("Please select a ticker symbol to use Showstopper mode.")



# ---------- Autopilot Investing Mode ----------
elif mode == "Autopilot Investing (Agent)":
    with st.container():
        st.markdown("<div class='section-header'>Autopilot Investing (Agent)</div>", unsafe_allow_html=True)

        # Load or initialize state
        state = load_state()

        # Left = controls (only first-time cash + risk), Right = portfolio view
        c1, c2 = st.columns([1,2])

        with c1:
            with st.container():
                st.markdown("**Portfolio Setup**")
                st.markdown("Configure your risk profile and initial investment amount.")
                
                risk = st.selectbox("Risk Profile", ["Low", "Medium", "High"], index=["low","medium","high"].index(state.get("risk_profile","medium")))
                initial_cash = st.number_input("Initial Cash (USD)", min_value=0.0, value=float(state.get("cash", 0.0)), step=100.0, help="If first run, set how much to invest. Later, adjust by adding cash here.")
                
                if st.button("Save & Start Autopilot", use_container_width=True, type="primary"):
                    # Save risk change and cash (additive if you want top-ups)
                    state["risk_profile"] = risk.lower()
                    # If brand new portfolio (no positions, no history), set cash = input
                    if not state.get("positions") and not state.get("history"):
                        state["cash"] = float(initial_cash)
                    else:
                        # Treat as a top-up if user increased it
                        delta = float(initial_cash) - float(state.get("cash", 0.0))
                        if delta > 0:
                            state["cash"] += delta
                    
                    with st.spinner("Setting up your portfolio..."):
                        # Import the functions we need for immediate investment
                        from agent.agent import portfolio_value
                        from agent.strategy import get_targets

                        # Ensure state containers exist
                        state.setdefault("positions", {})
                        state.setdefault("history", [])

                        # 1) Get target universe for risk
                        targets = get_targets(state["risk_profile"])  # e.g., {"AAPL":0.2, "MSFT":0.2, ...}
                        tickers = list(targets.keys())

                        # 2) Fetch robust last prices
                        import yfinance as yf
                        px = yf.download(tickers, period="5d", progress=False, auto_adjust=True, threads=True)

                        # Handle single vs multi-ticker frame; forward-fill; drop full-NaN rows
                        import pandas as pd
                        if hasattr(px.columns, "levels"):
                            last_row = px["Close"].ffill().dropna(how="all").iloc[-1]
                            last_prices = {t: float(v) for t, v in last_row.to_dict().items() if pd.notna(v) and v > 0}
                        else:
                            last_prices = {}
                            v = float(px["Close"].ffill().dropna().iloc[-1]) if not px.empty else None
                            if v and v > 0:
                                last_prices[tickers[0]] = v

                        # 3) Allocate with fractional shares (up to 4 decimals)
                        allocated_cash = 0.0
                        total_cash = float(state.get("cash", 0.0))
                        min_trade_dollars = 5.0  # avoid dust-size trades

                        for ticker, weight in targets.items():
                            price = last_prices.get(ticker)
                            if not price:
                                continue

                            target_dollars = total_cash * float(weight)
                            if target_dollars < min_trade_dollars:
                                continue

                            qty = round(target_dollars / price, 4)  # FRACTIONAL SHARES
                            cost = qty * price

                            if qty > 0 and (allocated_cash + cost) <= total_cash + 1e-9:
                                # accumulate position (support multiple runs)
                                current_qty = float(state["positions"].get(ticker, 0.0))
                                state["positions"][ticker] = round(current_qty + qty, 4)
                                allocated_cash += cost

                                state["history"].append({
                                    "ts": dt.datetime.now().timestamp(),
                                    "ticker": ticker,
                                    "side": "buy",
                                    "qty": qty,
                                    "price": price,
                                    "dollars": -round(cost, 2),
                                    "note": "initial allocation (fractional)"
                                })

                        # 4) Update cash, stamp last run, save
                        state["cash"] = round(total_cash - allocated_cash, 2)
                        state["last_run"] = dt.date.today().isoformat()
                        save_state(state)

                        # 5) User feedback
                        num_pos = sum(1 for q in state["positions"].values() if q > 0)
                        if num_pos > 0:
                            st.success(f"Autopilot started! Portfolio allocated with {num_pos} positions. The agent will now also run automatically once per day.")
                        else:
                            st.warning("Prices unavailable for your universe right now. Autopilot settings saved; the agent will try again on the daily run.")

        # Always auto-run (once per day) silently in the background
        state = maybe_auto_run(state)

        with c2:
            with st.container():
                st.markdown("**Portfolio Overview**")
                
                targets = get_targets(state["risk_profile"])
                # Get last known prices the agent used by re-reading (cheap, safe)
                tickers = list(targets.keys())
                try:
                    import yfinance as yf
                    px = yf.download(tickers, period="5d", progress=False)
                    if hasattr(px.columns, "levels"):
                        last_close = px["Close"].ffill().iloc[-1].to_dict()
                    else:
                        last_close = {tickers[0]: float(px["Close"].ffill().iloc[-1])}
                except Exception:
                    last_close = {}

                port_val = portfolio_value(state, last_close) if last_close else state.get("cash", 0.0)
                
                # Portfolio summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Portfolio Value", f"${port_val:,.2f}")
                with col2:
                    st.metric("Available Cash", f"${state.get('cash',0.0):,.2f}")
                with col3:
                    st.metric("Risk Profile", state.get('risk_profile','medium').title())

                # Positions table
                with st.container():
                    st.markdown("**Current Positions**")
                    
                    import pandas as pd
                    pos = state.get("positions", {})
                    if pos:
                        rows = []
                        for t, q in pos.items():
                            price = float(last_close.get(t, 0.0))
                            rows.append({
                                "Ticker": t,
                                "Shares": f"{float(q):.4f}",
                                "Last Price": f"${price:.2f}",
                                "Market Value": f"${float(q) * price:.2f}"
                            })
                        dfp = pd.DataFrame(rows).sort_values("Market Value", ascending=False)
                        
                        # Format the dataframe for better display
                        st.dataframe(
                            dfp, 
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("Click 'Save & Start Autopilot' to begin your portfolio.")

                # Recent activity
                with st.container():
                    st.markdown("**Recent Activity**")
                    
                    hist = state.get("history", [])
                    if hist:
                        last10 = hist[-10:]
                        for h in reversed(last10):
                            if "ticker" in h:
                                sign = "+" if h.get("side") == "sell" else "-"
                                dollars = h.get("dollars", 0.0)
                                side_color = "#56D4AA" if h.get("side") == "buy" else "#FF6B6B"  # Teal for buy, red for sell
                                st.markdown(
                                    f"""
                                    <div style="background: var(--panel); padding: 12px; border-radius: 8px; border-left: 4px solid {side_color}; margin: 8px 0; border: 1px solid var(--stroke); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                        <div style="color: var(--text); font-weight: 600; margin-bottom: 4px;">
                                            {dt.datetime.fromtimestamp(h['ts']).strftime('%Y-%m-%d %H:%M')}
                                        </div>
                                        <div style="color: {side_color}; font-weight: bold; font-size: 1.1em; margin-bottom: 4px;">
                                            {h['side'].upper()} {h['qty']} {h['ticker']} @ ${h['price']:.2f}
                                        </div>
                                        <div style="color: var(--muted); font-size: 0.9em;">
                                            {sign}${abs(dollars):,.2f} • {h.get('note','')}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown(
                                    f"""
                                    <div style="background: var(--panel); padding: 12px; border-radius: 8px; border-left: 4px solid var(--accent-purple); margin: 8px 0; border: 1px solid var(--stroke);">
                                        <div style="color: var(--text); font-weight: 600;">
                                            {dt.datetime.fromtimestamp(h.get('ts', 0)).strftime('%Y-%m-%d %H:%M')}
                                        </div>
                                        <div style="color: var(--muted);">
                                            {h.get('event', 'Unknown event')}
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No activity yet. Start your portfolio to see trading history.")


# ---------- Footer ----------
with st.container():
    st.markdown("---")
    st.markdown(
        """
    <div class="footer" style="text-align:center; padding: 20px; margin-top: 10px; border-top: 2px solid; border-image: linear-gradient(90deg, var(--accent-teal), var(--accent-purple)) 1;">
        <strong>Prophacy AI</strong> — Powered by Prophet + FinBERT Sentiment Analysis<br>
        <small>Disclaimer: This is a predictive model for educational purposes only. Not financial advice.</small>
    </div>
    """,
        unsafe_allow_html=True,)