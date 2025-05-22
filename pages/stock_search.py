import os
import random
import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
from bs4 import BeautifulSoup
import ta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# .env에서 API 키 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- 다크/라이트 모드 테마 정의 ---
THEME_COLORS = {
    "light": {
        "증가": {
            "candle": "#FF5252",     # 빨간색 양봉
            "avg_line": "#FF9E80",   # 주황색 계열 이동평균선
            "area": "rgba(255, 82, 82, 0.7)", # 면적 차트용 색상
            "line": "#FF5252",       # 선 차트용 색상
            "text": "#FF1744",       # 텍스트 색상
            "background": "#FFFFFF",  # 배경색
            "plot_bg": "rgba(255, 255, 255, 0.95)",
            "grid": "rgba(200, 200, 200, 0.3)",
            "text_color": "#353535"
        },
        "감소": {
            "candle": "#29B6F6",     # 파란색 음봉
            "avg_line": "#80D8FF",   # 하늘색 계열 이동평균선
            "area": "rgba(41, 182, 246, 0.7)", # 면적 차트용 색상
            "line": "#29B6F6",       # 선 차트용 색상
            "text": "#0091EA",       # 텍스트 색상
            "background": "#FFFFFF",  # 배경색
            "plot_bg": "rgba(255, 255, 255, 0.95)",
            "grid": "rgba(200, 200, 200, 0.3)",
            "text_color": "#353535"
        }
    },
    "dark": {
        "증가": {
            "candle": "#FF5252",     # 빨간색 양봉
            "avg_line": "#FF9E80",   # 주황색 계열 이동평균선
            "area": "rgba(255, 82, 82, 0.7)", # 면적 차트용 색상
            "line": "#FF5252",       # 선 차트용 색상
            "text": "#FF5252",       # 텍스트 색상
            "background": "#1E1E1E",  # 배경색
            "plot_bg": "rgba(30, 30, 30, 0.95)",
            "grid": "rgba(70, 70, 70, 0.3)",
            "text_color": "#FFFFFF"
        },
        "감소": {
            "candle": "#29B6F6",     # 파란색 음봉
            "avg_line": "#80D8FF",   # 하늘색 계열 이동평균선
            "area": "rgba(41, 182, 246, 0.7)", # 면적 차트용 색상
            "line": "#29B6F6",       # 선 차트용 색상
            "text": "#29B6F6",       # 텍스트 색상
            "background": "#1E1E1E",  # 배경색
            "plot_bg": "rgba(30, 30, 30, 0.95)",
            "grid": "rgba(70, 70, 70, 0.3)",
            "text_color": "#FFFFFF"
        }
    }
}

# --- 주식 데이터 캐싱 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, period='1y'):
    try:
        return yf.Ticker(ticker).history(period=period)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_stock_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

# --- 한국 주식 목록 확장 ---
@st.cache_data(ttl=3600*24)
def load_kr_stocks():
    # 기본 인기 종목
    stocks = {
        '삼성전자': '005930.KS', 'SK하이닉스': '000660.KS', 'LG에너지솔루션': '373220.KS',
        '삼성바이오로직스': '207940.KS', '삼성SDI': '006400.KS', '현대차': '005380.KS',
        '기아': '000270.KS', '네이버': '035420.KS', 'LG화학': '051910.KS',
        '삼성전자우': '005935.KS', '포스코홀딩스': '005490.KS', '카카오': '035720.KS',
        '셀트리온': '068270.KS', 'KB금융': '105560.KS', '신한지주': '055550.KS',
        'LG전자': '066570.KS', '현대모비스': '012330.KS', 'SK이노베이션': '096770.KS',
        'SK텔레콤': '017670.KS', 'LG생활건강': '051900.KS', '한국전력': '015760.KS',
        '삼성물산': '028260.KS', '카카오뱅크': '323410.KS', 'SK바이오사이언스': '302440.KS'
    }
    
    # 추가 종목 (KOSPI, KOSDAQ 주요 종목)
    additional_stocks = {
        '현대건설': '000720.KS', '롯데케미칼': '011170.KS', 'S-Oil': '010950.KS',
        '두산에너빌리티': '034020.KS', '고려아연': '010130.KS', '하나금융지주': '086790.KS',
        '한화솔루션': '009830.KS', '한온시스템': '018880.KS', '우리금융지주': '316140.KS',
        '기업은행': '024110.KS', '한국조선해양': '009540.KS', '한미사이언스': '008930.KS',
        '두산밥캣': '241560.KS', '카카오페이': '377300.KS', '크래프톤': '259960.KS',
        'SK바이오팜': '326030.KS', 'LG이노텍': '011070.KS', '엔씨소프트': '036570.KS',
        'CJ제일제당': '097950.KS', '삼성중공업': '010140.KS', '현대글로비스': '086280.KS',
        '삼성엔지니어링': '028050.KS', '한국항공우주': '047810.KS', '에코프로': '086520.KQ',
        '에코프로비엠': '247540.KQ', '셀트리온헬스케어': '091990.KQ', '씨젠': '096530.KQ',
        '펄어비스': '263750.KQ', '에이치엘비': '028300.KQ', '카카오게임즈': '293490.KQ'
    }
    
    stocks.update(additional_stocks)
    return stocks

# --- 미국 인기 주식 확장 ---
@st.cache_data(ttl=3600*24)
def load_us_stocks():
    # 기본 인기 종목
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK-B', 
        'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'BAC', 'MA', 'DIS', 'ADBE',
        'CRM', 'NFLX', 'INTC', 'VZ', 'KO', 'PFE', 'T', 'AMD', 'CSCO'
    ]
    
    # 추가 종목
    additional_tickers = [
        'GOOG', 'WMT', 'NKE', 'MCD', 'PYPL', 'UBER', 'ABNB', 'SBUX', 'GME', 'AMC',
        'TXN', 'QCOM', 'MU', 'TSM', 'BABA', 'NIO', 'PLTR', 'COIN', 'SNOW', 'ZM',
        'MRNA', 'PFE', 'JNJ', 'BMY', 'LLY', 'ABBV', 'CVX', 'XOM', 'BP', 'GS', 
        'F', 'GM', 'RIVN', 'LCID', 'TWTR', 'EA', 'ATVI'
    ]
    
    tickers.extend(additional_tickers)
    
    # 주요 ETF 추가
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VEA', 'VWO', 'GLD', 'SLV', 'USO']
    tickers.extend(etfs)
    
    # 중복 제거
    return list(set(tickers))

# 주식 목록 로드
KR_STOCK_MAP = load_kr_stocks()
US_POPULAR_TICKERS = load_us_stocks()

STOCK_KEYWORDS = ['주가', '가격', '시세', '티커', '주식', '종목', '차트', '정보', '뉴스']
TICKER_REGEX = re.compile(r'^[A-Za-z0-9\.\-]+$')

# --- 한글 기업 요약 추출 (네이버 금융 크롤링) ---
@st.cache_data(ttl=3600*24)
def get_kr_company_summary(ticker):
    if not ticker or not (ticker.endswith('.KS') or ticker.endswith('.KQ')):
        return None
        
    try:
        code = ticker.split('.')[0]
        url = f"https://finance.naver.com/item/main.naver?code={code}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 기업개요 추출
        summary_div = soup.select_one('#summary_info')
        if summary_div:
            return summary_div.get_text().strip()
            
        # 기업실적 분석 추출 (대안)
        analysis_div = soup.select_one('.corp_group1')
        if analysis_div:
            return analysis_div.get_text().strip()
            
        return None
    except:
        return None

# --- GPT를 이용한 기업 요약 번역 ---
def translate_with_gpt(text, target_lang='ko'):
    if not text or len(text) < 10:
        return ""
        
    try:
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4",  # GPT-4로 고정
            temperature=0.3,
            max_tokens=1000
        )
        prompt = f"다음 영문 텍스트를 자연스러운 한국어로 번역해주세요:\n\n{text}"
        return llm.predict(prompt)
    except Exception as e:
        return text  # 오류 시 원문 반환

def get_company_summary(ticker):
    # 한국 주식은 네이버 금융에서 한글 정보 가져오기
    if ticker.endswith('.KS') or ticker.endswith('.KQ'):
        korean_summary = get_kr_company_summary(ticker)
        if korean_summary:
            return korean_summary
    
    # 그 외에는 yfinance에서 정보 가져오고 번역
    try:
        info = fetch_stock_info(ticker)
        summary = info.get('longBusinessSummary', "")
        if summary and len(summary) > 50:
            # GPT로 번역
            return translate_with_gpt(summary)
        return ""
    except:
        return ""

def is_valid_ticker(sym):
    try:
        info = fetch_stock_info(sym)
        return bool(info.get('symbol'))
    except:
        return False

def extract_ticker_and_name(query):
    q = query.strip().upper() 
    
    # 1) 티커 형식이면 yfinance에서 회사명 추출
    if TICKER_REGEX.match(q) and is_valid_ticker(q):
        ticker = q.upper()
        info = fetch_stock_info(ticker)
        name = info.get('longName') or info.get('shortName') or ticker
        return ticker, name
        
    # 2) 한국 대표 종목명 매핑
    for name, tk in KR_STOCK_MAP.items():
        if name in query:
            return tk, name
            
    # 3) 미국 인기 종목 검색
    for ticker in US_POPULAR_TICKERS:
        if ticker.upper() in query.upper():
            info = fetch_stock_info(ticker)
            name = info.get('longName') or info.get('shortName') or ticker
            return ticker, name
            
    # 4) yfinance 검색
    try:
        info = fetch_stock_info(query)
        ticker = info.get('symbol')
        name = info.get('longName') or info.get('shortName') or query
        return ticker, name
    except:
        return None, query

# --- 고급 기술적 분석 지표 계산 ---
def calculate_technical_indicators(df):
    # 이동평균선
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # 볼린저 밴드
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Middle'] = bollinger.bollinger_mavg()
    
    # ATR (Average True Range) - 변동성 지표
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    return df

# --- 고급 주식 차트 시각화 ---
def plot_interactive_chart(ticker, name, period='1y'):
    # 주가 데이터 가져오기
    df = fetch_stock_data(ticker, period)
    if df.empty:
        st.error(f"⚠️ '{name}'({ticker}) 데이터가 없습니다.")
        return None, None
        
    # 기술적 분석 지표 계산
    df = calculate_technical_indicators(df)
    
    # 날짜 인덱스 처리
    df = df.reset_index()
    
    # 1. 캔들스틱 + 이동평균선 차트
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2], 
                       subplot_titles=(f"{name} 주가 차트", "거래량", "기술적 지표"))
    
    # 캔들스틱 차트 추가
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="주가",
            increasing_line_color='#FF4B4B',  # 양봉 색상
            decreasing_line_color='#1C86EE'   # 음봉 색상
        ),
        row=1, col=1
    )
    
    # 이동평균선 추가
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MA20'], name="20일 이동평균", line=dict(color='#FF8C00', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MA50'], name="50일 이동평균", line=dict(color='#9370DB', width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['MA200'], name="200일 이동평균", line=dict(color='#20B2AA', width=1.5)),
        row=1, col=1
    )
    
    # 볼린저 밴드 추가
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Upper'], name="볼린저 상단", line=dict(color='rgba(250,128,114,0.7)', width=1, dash='dash')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['BB_Lower'], name="볼린저 하단", line=dict(color='rgba(135,206,235,0.7)', width=1, dash='dash')),
        row=1, col=1
    )
    
    # 거래량 차트 추가
    colors = ['#FF4B4B' if row['Close'] > row['Open'] else '#1C86EE' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name="거래량", marker_color=colors),
        row=2, col=1
    )
    
    # RSI 추가 (기술적 지표)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='#9932CC', width=1.5)),
        row=3, col=1
    )
    
    # RSI 기준선 (30, 70)
    fig.add_trace(
        go.Scatter(x=df['Date'], y=[70] * len(df), name="RSI 70", line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dash')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=[30] * len(df), name="RSI 30", line=dict(color='rgba(0,128,0,0.5)', width=1, dash='dash')),
        row=3, col=1
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title=f"{name} ({ticker}) 주가 분석",
        height=800,  # 높이 증가
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
    )
    
    # 가격 표시 형식 설정
    fig.update_yaxes(tickprefix="", tickformat=",.0f", row=1, col=1)
    
    # 그리드 설정
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    # Y축 제목 설정
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    # 금융 지표 계산
    latest = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2] if len(df) > 1 else latest
    change = latest - prev
    pct = (change/prev*100) if prev else 0
    vol = df['Volume'].iloc[-1]
    
    # 추가 금융정보 가져오기
    info = fetch_stock_info(ticker)
    market_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 0)
    dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
    beta = info.get('beta', 0)
    
    # 지표 표시
    metrics = {
        "현재가": f"{latest:,.2f}",
        "전일 대비": f"{change:+,.2f} ({pct:+.2f}%)",
        "거래량": f"{int(vol):,}주",
        "52주 최고가": f"{df['High'].max():,.2f}",
        "52주 최저가": f"{df['Low'].min():,.2f}",
        "시가총액": f"{market_cap:,.0f}" if market_cap else "N/A",
        "P/E 비율": f"{pe_ratio:.2f}" if pe_ratio else "N/A",
        "배당수익률": f"{dividend_yield:.2f}%" if dividend_yield else "N/A",
        "베타": f"{beta:.2f}" if beta else "N/A"
    }
    
    return fig, metrics

# --- 추가 그래프: MACD 차트 ---
def plot_macd_chart(ticker, period='1y'):
    df = fetch_stock_data(ticker, period)
    if df.empty:
        return None
        
    df = calculate_technical_indicators(df)
    df = df.reset_index()
    
    fig = go.Figure()
    
    # MACD 라인과 시그널 라인
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            name="MACD",
            line=dict(color='#2171B5', width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MACD_Signal'],
            name="신호선",
            line=dict(color='#FB6A4A', width=1.5)
        )
    )
    
    # MACD 히스토그램
    colors = ['#2ECC71' if val > 0 else '#E74C3C' for val in df['MACD_Hist']]
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['MACD_Hist'],
            name="MACD 히스토그램",
            marker_color=colors
        )
    )
    
    fig.update_layout(
        title="MACD 지표 분석",
        height=400,
        hovermode="x unified",
        yaxis_title="MACD",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1)
    )
    
    return fig

# --- 추가 그래프: 변동성 분석 ---
def plot_volatility_chart(ticker, period='1y'):
    df = fetch_stock_data(ticker, period)
    if df.empty:
        return None
        
    df = calculate_technical_indicators(df)
    df = df.reset_index()
    
    # ATR 차트
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['ATR'],
            name="ATR (변동성)",
            line=dict(color='#8E44AD', width=2)
        )
    )
    
    # 20일 변동성 (표준편차)
    df['Volatility_20d'] = df['Close'].rolling(window=20).std()
    
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Volatility_20d'],
            name="20일 표준편차",
            line=dict(color='#F39C12', width=2)
        )
    )
    
    fig.update_layout(
        title="가격 변동성 분석",
        height=350,
        hovermode="x unified",
        yaxis_title="변동성",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1)
    )
    
    return fig

# --- 추가 그래프: 수익률 비교 차트 ---
def plot_return_comparison(ticker, period='1y'):
    # 대상 주식 데이터
    df = fetch_stock_data(ticker, period)
    if df.empty:
        return None
    
    # 시장 지수 데이터 (S&P 500 또는 KOSPI)
    is_korean = ticker.endswith('.KS') or ticker.endswith('.KQ')
    market_ticker = '^KS11' if is_korean else '^GSPC'  # KOSPI 또는 S&P 500
    
    market_df = fetch_stock_data(market_ticker, period)
    if market_df.empty:
        return None
    
    # 수익률 계산
    df = df.reset_index()
    market_df = market_df.reset_index()
    
    # 시작 날짜 정렬
    start_date = max(df['Date'].min(), market_df['Date'].min())
    df = df[df['Date'] >= start_date]
    market_df = market_df[market_df['Date'] >= start_date]
    
    # 수익률 계산
    df['return'] = df['Close'] / df['Close'].iloc[0] - 1
    market_df['return'] = market_df['Close'] / market_df['Close'].iloc[0] - 1
    
    # 결합 데이터 생성
    comparison_df = pd.DataFrame({
        'Date': df['Date'],
        'stock_return': df['return'] * 100,  # 퍼센트로 변환
        'market_return': market_df['return'].reindex(index=df.index, method='ffill') * 100  # 퍼센트로 변환
    })
    
    # 차트 생성
    fig = go.Figure()
    
    stock_name = ticker.split('.')[0] if '.' in ticker else ticker
    market_name = 'KOSPI' if is_korean else 'S&P 500'
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['stock_return'],
            name=f"{stock_name} 수익률",
            line=dict(color='#E74C3C', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Date'],
            y=comparison_df['market_return'],
            name=f"{market_name} 수익률",
            line=dict(color='#3498DB', width=2)
        )
    )
    
    # 0% 선 추가
    fig.add_trace(
        go.Scatter(
            x=comparison_df['Date'],
            y=[0] * len(comparison_df),
            name="0% 기준선",
            line=dict(color='black', width=1, dash='dash')
        )
    )
    
    fig.update_layout(
        title=f"{stock_name} vs {market_name} 수익률 비교",
        height=350,
        hovermode="x unified",
        yaxis_title="수익률 (%)",
        yaxis_tickformat='.1f',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1)
    )
    
    return fig

# --- 분기별 실적 차트 ---
def plot_financial_stats(ticker):
    try:
        # 재무 데이터 가져오기
        stock = yf.Ticker(ticker)
        
        # 분기별 매출 및 순이익
        earnings = stock.quarterly_earnings
        if earnings is not None and not earnings.empty:
            earnings = earnings.reset_index()
            
            # 매출 및 순이익 차트
            fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Bar(
                    x=earnings['Year'], 
                    y=earnings['Revenue'], 
                    name="분기별 매출",
                    marker_color='rgba(0, 128, 255, 0.7)'
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Scatter(
                    x=earnings['Year'], 
                    y=earnings['Earnings'], 
                    name="분기별 순이익",
                    marker_color='rgba(255, 69, 0, 0.9)',
                    mode='lines+markers'
                ),
                secondary_y=True,
            )
            
            fig.update_layout(
                title="분기별 실적 추이",
                xaxis_title="분기",
                yaxis_title="매출 (백만 달러)",
                yaxis2_title="순이익 (백만 달러)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            
            st.plotly_chart(fig, theme=None, use_container_width=True)
            
        # 주요 재무지표 추출
        info = fetch_stock_info(ticker)
        
        # 재무지표 표시
        st.markdown("### 주요 재무지표")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("ROE (자기자본이익률)", f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A")
            st.metric("매출총이익률", f"{info.get('grossMargins', 0) * 100:.2f}%" if info.get('grossMargins') else "N/A")
        
        with col2:
            st.metric("ROA (총자산이익률)", f"{info.get('returnOnAssets', 0) * 100:.2f}%" if info.get('returnOnAssets') else "N/A")
            st.metric("영업이익률", f"{info.get('operatingMargins', 0) * 100:.2f}%" if info.get('operatingMargins') else "N/A")
        
        with col3:
            st.metric("부채비율", f"{info.get('debtToEquity', 0):.2f}%" if info.get('debtToEquity') else "N/A")
            st.metric("순이익률", f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get('profitMargins') else "N/A")
            
    except Exception as e:
        st.warning(f"재무정보를 가져오는 중 오류가 발생했습니다: {e}")

# --- 동종업계 종목 추천 및 비교 ---
@st.cache_data(ttl=3600)
def find_related_stocks(ticker, limit=4):
    try:
        info = fetch_stock_info(ticker)
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        
        if not (sector or industry):
            return {}
            
        # 한국 주식인지 확인
        is_korean = '.KS' in ticker or '.KQ' in ticker
        
        result = {}
        
        if is_korean:
            # 한국 주식은 KR_STOCK_MAP에서 검색
            for name, tk in KR_STOCK_MAP.items():
                if tk != ticker:  # 현재 종목 제외
                    try:
                        stock_info = fetch_stock_info(tk)
                        if stock_info.get('sector') == sector or stock_info.get('industry') == industry:
                            result[name] = tk
                            if len(result) >= limit:
                                break
                    except:
                        continue
        else:
            # 미국 주식은 US_POPULAR_TICKERS에서 검색
            for tk in US_POPULAR_TICKERS:
                if tk != ticker:  # 현재 종목 제외
                    try:
                        stock_info = fetch_stock_info(tk)
                        if stock_info.get('sector') == sector or stock_info.get('industry') == industry:
                            name = stock_info.get('longName', tk)
                            result[name] = tk
                            if len(result) >= limit:
                                break
                    except:
                        continue
        
        # 결과가 없으면 랜덤 종목 추가
        if not result and is_korean:
            random_tickers = random.sample(list(KR_STOCK_MAP.items()), min(limit, len(KR_STOCK_MAP)))
            result = {name: tk for name, tk in random_tickers if tk != ticker}
        elif not result:
            random_tickers = random.sample(US_POPULAR_TICKERS, min(limit, len(US_POPULAR_TICKERS)))
            result = {fetch_stock_info(tk).get('longName', tk): tk for tk in random_tickers if tk != ticker}
            
        return result
    except:
        return {}

def display_stock_recommendation(ticker):
    st.markdown("### 관련 종목 추천")
    
    related_stocks = find_related_stocks(ticker)
    
    if related_stocks:
        cols = st.columns(len(related_stocks))
        
        for i, (name, related_ticker) in enumerate(related_stocks.items()):
            with cols[i]:
                try:
                    df = fetch_stock_data(related_ticker, period='1mo')
                    latest = df['Close'].iloc[-1] if not df.empty and len(df) > 0 else 0
                    pct_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100 if not df.empty and len(df) > 1 else 0
                    
                    # 간단한 스파크라인 차트
                    fig = px.line(df, x=df.index, y='Close', title=f"{name}")
                    fig.update_layout(height=100, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                    fig.update_xaxes(visible=False)
                    
                    st.plotly_chart(fig, theme=None, use_container_width=True)
                    st.metric(f"{related_ticker}", f"{latest:.2f}", f"{pct_change:.1f}%")
                except:
                    st.write(f"{name} ({related_ticker})")
                    st.warning("데이터 없음")
    else:
        st.info("관련 종목을 찾을 수 없습니다.")

# --- 개선된 트레이더 뷰 (3x3 고정, 다크/라이트 모드만 지원) ---
def display_trader_view(tickers_list, period='1mo'):
    st.markdown("### 🖥️ 트레이더 뷰: 다중 종목 모니터링")
    
    # 보기 옵션 (차트 스타일만 남기고 열 수 고정)
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # 다크모드/라이트모드 선택
        is_dark_mode = st.toggle("다크 모드", value=False)
    
    with col2:
        # 차트 스타일 선택
        chart_style = st.selectbox(
            "차트 스타일", 
            ["캔들스틱", "선 차트", "면적 차트"]
        )
    
    # 테마 설정 (다크/라이트 모드)
    theme_mode = "dark" if is_dark_mode else "light"
    
    # 고정 열 수 (3x3)
    cols_per_row = 3
    
    # 총 종목 수에 따라 필요한 종목 추출 (최대 9개)
    tickers_to_show = tickers_list[:9]
    total_tickers = len(tickers_to_show)
    
    # 필요한 행 수 계산
    rows_needed = (total_tickers + cols_per_row - 1) // cols_per_row
    
    # 각 행에 대해 처리
    for row in range(rows_needed):
        # 각 행마다 columns 생성
        cols = st.columns(cols_per_row)
        
        # 각 열에 차트 배치
        for col_idx in range(cols_per_row):
            ticker_idx = row * cols_per_row + col_idx
            
            # 인덱스 범위 체크
            if ticker_idx < total_tickers:
                ticker, name = tickers_to_show[ticker_idx]
                
                with cols[col_idx]:
                    try:
                        df = fetch_stock_data(ticker, period)
                        if not df.empty:
                            # 현재가 및 변동률 계산
                            latest = df['Close'].iloc[-1]
                            prev = df['Close'].iloc[-2] if len(df) > 1 else latest
                            change = latest - prev
                            pct = (change/prev*100) if prev else 0
                            
                            # 증가/감소에 따라 색상 선택
                            price_direction = "증가" if change >= 0 else "감소"
                            color_set = THEME_COLORS[theme_mode][price_direction]
                            
                            # 차트 생성
                            fig = go.Figure()
                            
                            if chart_style == "캔들스틱":
                                # 캔들스틱 차트
                                fig.add_trace(
                                    go.Candlestick(
                                        x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name="OHLC",
                                        showlegend=False,
                                        increasing_line_color=THEME_COLORS[theme_mode]["증가"]["candle"],
                                        decreasing_line_color=THEME_COLORS[theme_mode]["감소"]["candle"],
                                        increasing_fillcolor=THEME_COLORS[theme_mode]["증가"]["candle"],
                                        decreasing_fillcolor=THEME_COLORS[theme_mode]["감소"]["candle"]
                                    )
                                )
                                
                            elif chart_style == "면적 차트":
                                # 면적 차트
                                fig.add_trace(
                                    go.Scatter(
                                        x=df.index,
                                        y=df['Close'],
                                        name="주가",
                                        fill='tozeroy',
                                        fillcolor=color_set["area"],
                                        line=dict(color=color_set["candle"], width=2)
                                    )
                                )
                                
                            else:  # 선 차트
                                # 선 차트
                                fig.add_trace(
                                    go.Scatter(
                                        x=df.index,
                                        y=df['Close'],
                                        name="주가",
                                        line=dict(color=color_set["line"], width=2)
                                    )
                                )
                            
                            # 20일 이동평균선 추가
                            if len(df) >= 20:
                                ma20 = df['Close'].rolling(window=20).mean()
                                fig.add_trace(
                                    go.Scatter(
                                        x=df.index, 
                                        y=ma20, 
                                        name="MA20",
                                        line=dict(color=color_set["avg_line"], width=1.5, dash='dot'),
                                        showlegend=False
                                    )
                                )
                            
                            # 레이아웃 설정 (테마에 따라 다르게)
                            fig.update_layout(
                                title=dict(
                                    text=f"{name} ({ticker})",
                                    font=dict(size=16, family="Arial", color=color_set["text_color"]),
                                    x=0.5,
                                    y=0.98,
                                    xanchor='center',
                                    yanchor='top'
                                ),
                                height=250,  # 3x3 그리드에 맞게 높이 조정
                                margin=dict(l=0, r=0, t=30, b=0),
                                xaxis_rangeslider_visible=False,
                                xaxis=dict(
                                    showgrid=False,
                                    showticklabels=True,
                                    linecolor=color_set["grid"],
                                    color=color_set["text_color"]
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor=color_set["grid"],
                                    showticklabels=True,
                                    tickformat=",.0f",
                                    color=color_set["text_color"]
                                ),
                                plot_bgcolor=color_set["plot_bg"],
                                paper_bgcolor=color_set["background"]
                            )
                            
                            # Y축 표시 형식 설정
                            fig.update_yaxes(tickprefix="", tickformat=",.0f")
                            
                            # 차트 표시 (테마 없이 원본 색상 유지)
                            st.plotly_chart(fig, theme=None, use_container_width=True)
                            
                            # 가격과 변동률 표시
                            st.markdown(
                                f"""<div style="text-align: center;">
                                   <span style="font-size: 18px; font-weight: bold; color: {color_set["text"]};">
                                      {"📈" if change >= 0 else "📉"} {latest:,.2f}
                                   </span><br>
                                   <span style="color: {color_set["text"]};">
                                      {change:+,.2f} ({pct:+.2f}%)
                                   </span>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                            
                        else:
                            # 데이터 없음 표시
                            st.error(f"{name} 데이터 없음")
                            
                    except Exception as e:
                        # 오류 표시
                        st.error(f"{name} 오류: {str(e)}")

def get_llm_response(query):
    # GPT-4로 고정
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",  # 모델을 GPT-4로 고정
        temperature=0.3,
        max_tokens=1000,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # 한국어 응답 유도
    prompt = f"다음 질문에 한국어로 답변해주세요. 주식과 금융 관련 질문입니다: {query}"
    return llm.predict(prompt)

def render_stock_search():
    # 채팅 메시지 초기화
    st.session_state.setdefault("stock_msgs", [
        {"role":"assistant", 
         "content":"회사명이나 티커를 입력하면 실시간 차트와 지표를 보여드립니다. 일반 금융 질문도 답변해 드려요!"}
    ])
    st.session_state.setdefault("init_shown", False)

    # 인기 종목 트레이더 뷰 (상단에 배치)
    st.markdown("## 인기 종목 모니터링")
    
    # 국가 선택
    market = st.radio("시장 선택", ["한국", "미국"], horizontal=True)
    
    if market == "한국":
        # 한국 인기 종목
        popular_kr = [
            ('005930.KS', '삼성전자'), 
            ('000660.KS', 'SK하이닉스'),
            ('035420.KS', '네이버'),
            ('035720.KS', '카카오'),
            ('051910.KS', 'LG화학'),
            ('207940.KS', '삼성바이오'),
            ('005380.KS', '현대차'),
            ('000270.KS', '기아'),
            ('068270.KS', '셀트리온'),
        ]
        display_trader_view(popular_kr)
        
    else:
        # 미국 인기 종목
        popular_us = [
            ('AAPL', 'Apple'), 
            ('MSFT', 'Microsoft'),
            ('GOOGL', 'Alphabet'),
            ('AMZN', 'Amazon'),
            ('META', 'Meta'),
            ('TSLA', 'Tesla'),
            ('NVDA', 'NVIDIA'),
            ('JPM', 'JPMorgan'),
            ('V', 'Visa'),
        ]
        display_trader_view(popular_us)
        
    # 커스텀 종목 추가 옵션
    with st.expander("커스텀 종목 모니터링"):
        custom_input = st.text_input("종목 티커를 쉼표로 구분하여 입력하세요 (최대 9개 표시)", 
                                     placeholder="AAPL,MSFT,GOOGL 또는 005930.KS,035420.KS")
        
        if custom_input:
            custom_tickers = [ticker.strip() for ticker in custom_input.split(',')][:9]  # 최대 9개로 제한
            custom_ticker_info = []
            
            for ticker in custom_tickers:
                if ticker:
                    try:
                        info = fetch_stock_info(ticker)
                        name = info.get('longName', ticker) or info.get('shortName', ticker) or ticker
                        custom_ticker_info.append((ticker, name))
                    except:
                        custom_ticker_info.append((ticker, ticker))
            
            if custom_ticker_info:
                st.markdown("### 커스텀 종목 차트")
                display_trader_view(custom_ticker_info)
    
    # 구분선
    st.markdown("---")
    
    # 단일 종목 상세 분석 (하단에 배치)
    st.markdown("## 종목 상세 분석")
    
    # 초기 랜덤 차트
    if not st.session_state.init_shown:
        name, ticker = random.choice(list(KR_STOCK_MAP.items()))
        fig, metrics = plot_interactive_chart(ticker, name)
        summary = get_company_summary(ticker)
        
        if fig:
            st.plotly_chart(fig, theme=None, use_container_width=True)
            
            # 지표를 3열로 표시
            metric_cols = st.columns(3)
            for i, (lbl, val) in enumerate(metrics.items()):
                col_idx = i % 3
                metric_cols[col_idx].metric(lbl, val)
            
            # 회사 설명
            if summary:
                st.markdown("### 기업 요약")
                st.write(summary)
            
            # 추가 차트 탭
            additional_tabs = st.tabs(["MACD 분석", "변동성 분석", "수익률 비교"])
            
            with additional_tabs[0]:
                macd_fig = plot_macd_chart(ticker)
                if macd_fig:
                    st.plotly_chart(macd_fig, theme=None, use_container_width=True)
                
            with additional_tabs[1]:
                vol_fig = plot_volatility_chart(ticker)
                if vol_fig:
                    st.plotly_chart(vol_fig, theme=None, use_container_width=True)
                
            with additional_tabs[2]:
                return_fig = plot_return_comparison(ticker)
                if return_fig:
                    st.plotly_chart(return_fig, theme=None, use_container_width=True)
            
            # 재무 정보 및 추천 종목
            with st.expander("📊 상세 재무정보 보기"):
                plot_financial_stats(ticker)
                
            display_stock_recommendation(ticker)
                
        st.session_state.init_shown = True

    # 이전 메시지 표시
    for msg in st.session_state.stock_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력
    query = st.chat_input("회사명, 티커 또는 주식 관련 질문을 입력하세요 (예: 삼성전자, AAPL, 005930.KS)")
    if query:
        st.session_state.stock_msgs.append({"role":"user","content":query})
        with st.chat_message("assistant"):
            ticker, name = extract_ticker_and_name(query)
            if ticker:
                fig, metrics = plot_interactive_chart(ticker, name)
                summary = get_company_summary(ticker)
                
                if fig:
                    st.plotly_chart(fig, theme=None, use_container_width=True)
                    
                    # 지표를 3열로 표시
                    metric_cols = st.columns(3)
                    for i, (lbl, val) in enumerate(metrics.items()):
                        col_idx = i % 3
                        metric_cols[col_idx].metric(lbl, val)
                    
                    # 회사 설명
                    if summary:
                        st.markdown("### 기업 요약")
                        st.write(summary)
                    
                    # 추가 차트 탭
                    additional_tabs = st.tabs(["MACD 분석", "변동성 분석", "수익률 비교"])
                    
                    with additional_tabs[0]:
                        macd_fig = plot_macd_chart(ticker)
                        if macd_fig:
                            st.plotly_chart(macd_fig, theme=None, use_container_width=True)
                        
                    with additional_tabs[1]:
                        vol_fig = plot_volatility_chart(ticker)
                        if vol_fig:
                            st.plotly_chart(vol_fig, theme=None, use_container_width=True)
                        
                    with additional_tabs[2]:
                        return_fig = plot_return_comparison(ticker)
                        if return_fig:
                            st.plotly_chart(return_fig, theme=None, use_container_width=True)
                    
                    # 재무 정보
                    with st.expander("📊 상세 재무정보 보기"):
                        plot_financial_stats(ticker)
                        
                    # 관련 종목 추천
                    display_stock_recommendation(ticker)
                    
                response = f"'{name}'({ticker}) 정보를 표시했습니다. 다른 종목이나 질문이 있으시면 입력해주세요."
            else:
                try:
                    response = get_llm_response(query)
                except Exception as e:
                    response = f"AI 응답 오류: {e}"
            
            st.markdown(response)
            st.session_state.stock_msgs.append({"role":"assistant","content":response})
