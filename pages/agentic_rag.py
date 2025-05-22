import os
import time
import streamlit as st
import re
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import numpy as np
import functools
import threading
import concurrent.futures

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- 고급 투자 프롬프트 ---
INVESTMENT_PROMPTS = {
    "💹 최적 자산배분 분석": "내 투자성향과 시장 상황을 고려하여 주식, 채권, 현금, 원자재, 부동산 등 자산군별 최적 배분 비율을 추천해주고, 각 자산군의 장단점과 투자 전략을 설명해줘.",
    "📈 성장주 포트폴리오": "향후 5년간 높은 성장이 예상되는 혁신 기술(AI, 로봇공학, 클린에너지, 우주산업) 관련 유망 성장주 5-7개를 추천하고, 각 기업의 경쟁우위와 성장 전망을 분석해줘.",
    "💰 배당주 투자전략": "안정적인 배당수익을 제공하는 우량 배당주 포트폴리오를 추천하고, 배당 성장률과 배당 지속가능성 측면에서 분석해줘. 배당주 투자의 장단점도 함께 설명해줘.",
    "🛡️ 방어적 포트폴리오": "경기침체나 시장 하락기에 상대적으로 안정적인 방어적 포트폴리오를 구성해줘. 저변동성, 고배당, 가치주 중심으로 구성하고 위험 관리 전략도 설명해줘.",
    "🌎 글로벌 분산투자": "지역별, 국가별로 균형 있게 분산된 글로벌 ETF 포트폴리오를 추천하고, 각 지역의 경제 전망과 투자 매력도를 설명해줘. 환율 리스크 관리 방법도 포함해줘.",
    "📊 삼성전자 심층분석": "삼성전자(005930.KS)의 현재 밸류에이션, 반도체 시장 전망, 경쟁사 대비 강점, 향후 5년 성장 동력을 분석하고, 투자 적합성을 평가해줘.",
    "🏢 애플 기업분석": "애플(AAPL)의 최근 실적, 성장 동력, 경쟁 환경, 밸류에이션을 심층 분석하고, 장기 투자 관점에서 투자 적합성을 평가해줘."
}

# --- 예쁜 색상 팔레트 정의 ---
COLOR_PALETTES = {
    "vibrant": px.colors.qualitative.Bold,
    "pastel": px.colors.qualitative.Pastel,
    "dark": px.colors.qualitative.Dark24,
    "light": px.colors.qualitative.Light24,
    "vivid": px.colors.qualitative.Vivid
}

# --- 차트 스타일 통일 함수 ---
def apply_chart_style(fig, title=None, height=450):
    if title:
        fig.update_layout(
            title={
                'text': title,
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial', 'color': '#353535'}
            },
            height=height,
            font=dict(family="Arial, sans-serif", size=14, color="#353535"),
            xaxis=dict(
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='rgba(200,200,200,0.8)',
                showticklabels=True,
                title_font=dict(color="#353535")
            ),
            yaxis=dict(
                showgrid=True, 
                gridwidth=0.5, 
                gridcolor='rgba(200,200,200,0.8)',
                showticklabels=True,
                title_font=dict(color="#353535")
            ),
            plot_bgcolor='rgba(245,245,245,0.9)',
            paper_bgcolor='rgba(245,245,245,0.9)',
            margin=dict(l=20, r=20, t=70, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(255,255,255,0.9)',
                font=dict(color="#353535")
            )
        )
    return fig

# --- 주식 데이터 캐싱 함수 ---
@functools.lru_cache(maxsize=32)
def get_stock_data(ticker, period='1y'):
    try:
        info = yf.Ticker(ticker).info
        history = yf.Ticker(ticker).history(period=period)
        return info, history
    except Exception as e:
        st.warning(f"티커 '{ticker}' 데이터 로드 실패: {e}")
        return {}, pd.DataFrame()

# --- 고급 주식 차트 시각화 ---
def plot_advanced_stock_chart(ticker):
    try:
        info, df = get_stock_data(ticker)
        name = info.get('longName', ticker)
        
        if df.empty:
            return f"'{ticker}' 데이터가 없습니다."
        
        # 1. 주가 + 거래량 차트 (서브플롯)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3],
                            subplot_titles=(f"{name} 주가 추이", "거래량"))
        
        # 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC",
                increasing_line=dict(color='#26a69a'),
                decreasing_line=dict(color='#ef5350')
            ),
            row=1, col=1
        )
        
        # 20일, 50일 이동평균선
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()
        
        fig.add_trace(
            go.Scatter(x=df.index, y=ma20, line=dict(color='rgba(255, 207, 102, 0.8)', width=2), name="20일 이동평균"),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=ma50, line=dict(color='rgba(83, 123, 255, 0.8)', width=2), name="50일 이동평균"),
            row=1, col=1
        )
        
        # 거래량 바 차트
        colors = ['#26a69a' if df.Close[i] > df.Close[i-1] else '#ef5350' for i in range(1, len(df.Close))]
        colors.insert(0, '#888888')  # 첫 번째 데이터 포인트의 색상
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name="거래량"),
            row=2, col=1
        )
        
        # 레이아웃 설정
        fig.update_layout(
            title=f"{name} (티커: {ticker}) 주가 분석",
            height=700,
            xaxis_rangeslider_visible=False,
            plot_bgcolor='rgba(250,250,250,0.9)',
            paper_bgcolor='rgba(250,250,250,0.9)',
            font=dict(family="Arial, sans-serif", size=14, color="#353535"),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                font=dict(color="#353535")
            )
        )
        
        fig.update_xaxes(gridcolor='rgba(200,200,200,0.8)', zeroline=False, row=1, col=1)
        fig.update_xaxes(gridcolor='rgba(200,200,200,0.8)', zeroline=False, row=2, col=1)
        fig.update_yaxes(gridcolor='rgba(200,200,200,0.8)', zeroline=False, row=1, col=1)
        fig.update_yaxes(gridcolor='rgba(200,200,200,0.8)', zeroline=False, row=2, col=1)
        
        # 테마 없이 원본 색상 유지
        st.plotly_chart(fig, theme=None, use_container_width=True)
        
        # 2. 기본 정보 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("현재가", f"${df['Close'].iloc[-1]:.2f}", f"{((df['Close'].iloc[-1]/df['Close'].iloc[-2])-1)*100:.2f}%")
        with col2:
            st.metric("52주 최고", f"${df['High'].max():.2f}")
        with col3:
            st.metric("52주 최저", f"${df['Low'].min():.2f}")
        with col4:
            st.metric("평균 거래량", f"{int(df['Volume'].mean()):,}")
        
        # 3. 수익률 성과 분석
        if len(df) > 30:
            returns = df['Close'].pct_change().dropna()
            daily_return = returns.mean() * 100
            monthly_return = (((1 + returns.mean()) ** 21) - 1) * 100
            volatility = returns.std() * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("일일 평균 수익률", f"{daily_return:.3f}%")
            with col2:
                st.metric("월간 예상 수익률", f"{monthly_return:.2f}%")
            with col3:
                st.metric("일일 변동성", f"{volatility:.2f}%")
        
        return ""
    except Exception as e:
        return f"{ticker} 차트 생성 오류: {e}"

# --- 문서 자동 로드 ---
@st.cache_resource(ttl="1h")
def load_predefined_documents():
    DOCUMENTS_PATH = "./documents"
    PDF_FILES = ["pdf1.pdf", "pdf2.pdf", "pdf3.pdf"]
    
    docs = []
    for filename in PDF_FILES:
        filepath = os.path.join(DOCUMENTS_PATH, filename)
        if os.path.exists(filepath):
            try:
                docs += PyPDFLoader(filepath).load_and_split()
            except:
                pass
    
    if not docs:
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def rag_search(query: str, vectorstore, llm) -> str:
    if not vectorstore:
        return "참고 문서가 없습니다."
    
    retriever = vectorstore.as_retriever(search_kwargs={"k":2})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    return chain.run(query)

# --- 포트폴리오 분석 Tool ---
def analyze_portfolio(survey, llm) -> str:
    desc = get_portfolio_description(survey)
    prompt = f"""아래는 사용자의 포트폴리오 설문 결과입니다.
{desc}
이 조건에 맞는 최적의 포트폴리오(종목, 비중, 국가, 업종, 자산군 등)를 추천하고,
추천 이유, 리스크 요인, 분산 효과, 업종별 전망, 자산군별 전략을 설명해줘.
포트폴리오는 마크다운 표로, 설명은 자연어로 출력해줘.

반드시 종목명(Name), 티커(Ticker), 비중(Weight), 국가(Country), 업종(Industry) 컬럼이 있는 표 형태로 작성해줘."""
    return llm.predict(prompt)

# --- 설문 UI ---
def portfolio_survey():
    st.markdown("### 📝 내 주식 현황 및 투자성향 설문")
    knowledge = st.selectbox("포트폴리오 관리에 대한 이해도", ["전혀 없음", "기초적", "보통", "상당히 높음"])
    purpose = st.selectbox("투자 목적", ["은퇴자금", "단기수익", "자녀교육", "주택구입", "자산증식", "기타"])
    sector = st.multiselect("선호 업종", ["IT/테크", "헬스케어", "금융", "에너지", "소비재", "부동산", "대체투자", "기타"], default=["IT/테크"])
    risk = st.radio("투자 성향", ["안정형", "중립형", "공격형"])
    period = st.selectbox("예상 투자 기간", ["1년 미만", "1~3년", "3~5년", "5년 이상"])
    region = st.multiselect("관심 국가/지역", ["한국", "미국", "중국", "일본", "유럽", "신흥국", "기타"], default=["한국","미국"])
    asset_types = st.multiselect("현재 보유 자산군", ["주식", "채권", "현금", "부동산", "대체투자(금,원자재 등)", "암호화폐"], default=["주식"])
    rebalance = st.selectbox("포트폴리오 리밸런싱 주기", ["1개월", "3개월", "6개월", "1년", "필요시", "안함"])
    esg = st.radio("ESG/지속가능 투자 관심도", ["매우 높음", "관심 있음", "중립", "관심 없음"])
    alt_inv = st.radio("대체투자(금, 원자재, 암호화폐 등) 선호도", ["적극적", "일부", "관심 없음"])
    tickers = st.text_input("주요 투자 종목(티커, 콤마로 구분)", placeholder="예: AAPL, TSLA, 005930.KS")
    amount = st.slider("총 투자금(만원)", 100, 10000, 1000, 100)
    return {
        "knowledge": knowledge,
        "purpose": purpose,
        "sector": sector,
        "risk": risk,
        "period": period,
        "region": region,
        "asset_types": asset_types,
        "rebalance": rebalance,
        "esg": esg,
        "alt_inv": alt_inv,
        "tickers": tickers,
        "amount": amount
    }

def get_portfolio_description(survey):
    desc = (
        f"포트폴리오 이해도: {survey['knowledge']}\n"
        f"투자 목적: {survey['purpose']}\n"
        f"선호 업종: {', '.join(survey['sector'])}\n"
        f"투자 성향: {survey['risk']}\n"
        f"투자 기간: {survey['period']}\n"
        f"관심 국가/지역: {', '.join(survey['region'])}\n"
        f"현재 보유 자산군: {', '.join(survey['asset_types'])}\n"
        f"리밸런싱 주기: {survey['rebalance']}\n"
        f"ESG/지속가능 투자 관심도: {survey['esg']}\n"
        f"대체투자 선호도: {survey['alt_inv']}\n"
        f"주요 투자 종목(티커): {survey['tickers']}\n"
        f"총 투자금: {survey['amount']}만원"
    )
    return desc

# --- 마크다운 표 파싱 및 시각화 ---
def extract_markdown_table(answer):
    lines = answer.splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if "|" in line:
            table_lines.append(line)
            in_table = True
        elif in_table and line.strip() == "":
            break
    if not table_lines:
        return None, answer
    table_md = "\n".join(table_lines)
    pre = answer.split(table_md)[0].strip()
    post = answer.split(table_md)[1].strip() if table_md in answer else ""
    return table_md, pre + "\n\n" + post

def parse_portfolio_table(table_md):
    try:
        lines = table_md.strip().splitlines()
        clean_lines = [
            line for line in lines
            if not (set(line.replace('|', '').replace(' ', '')) <= {'-'})
        ]
        clean_table_md = "\n".join(clean_lines)
        
        buffer = StringIO(clean_table_md)
        df = pd.read_csv(buffer, sep="|", skipinitialspace=True)
        
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = df.columns.str.strip()
        
        weight_col = None
        for c in df.columns:
            if any(keyword in c.lower() for keyword in ['weight', '비중', 'ratio']):
                weight_col = c
                break
        
        if not weight_col:
            return None
            
        # 비중 값 정규화
        df[weight_col] = df[weight_col].astype(str).str.replace('%', '').str.replace(',', '')
        df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
        df = df.dropna(subset=[weight_col]).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        st.error(f"포트폴리오 표 파싱 오류: {str(e)}")
        return None

def plot_portfolio_interactive(df):
    # 기본 열 설정
    ticker_col = df.columns[0]
    
    # 비중 열 찾기
    weight_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['weight', '비중', 'ratio'])]
    if not weight_cols:
        weight_col = df.columns[1]
    else:
        weight_col = weight_cols[0]
    
    # 탭 인터페이스로 다양한 시각화 구성
    tabs = st.tabs(["포트폴리오 구성", "위험-수익 분석", "상관관계", "미래 시뮬레이션", "지역 분포"])
    
    with tabs[0]:
        # 1. 종목별 비중 도넛 차트 (기존 시각화)
        st.subheader("종목별 포트폴리오 구성")
        sorted_df = df.sort_values(by=weight_col, ascending=False).reset_index(drop=True)
        
        fig1 = px.pie(
            sorted_df, 
            values=weight_col, 
            names=ticker_col, 
            title="종목별 비중",
            hole=0.4,
            color_discrete_sequence=COLOR_PALETTES['vivid']
        )
        
        fig1.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont=dict(size=14, color='black'),
            insidetextfont=dict(color='white'),
            outsidetextfont=dict(color='black'),
            hovertemplate='<b>%{label}</b><br>비중: %{percent:.1%}<br>값: %{value:.1f}%'
        )
        
        apply_chart_style(fig1, "🔮 종목별 투자 비중")
        st.plotly_chart(fig1, theme=None, use_container_width=True)
        
        # 국가별/업종별 비중 (기존 코드)
        country_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['country', '국가', 'region'])]
        if country_cols:
            country_col = country_cols[0]
            country_df = df.groupby(country_col)[weight_col].sum().sort_values(ascending=False).reset_index()
            
            fig2 = px.bar(
                country_df,
                x=country_col,
                y=weight_col,
                title="국가별 비중",
                text=weight_col,
                color=country_col,
                color_discrete_sequence=COLOR_PALETTES['dark']
            )
            
            fig2.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=14, color='black'),
                hovertemplate='<b>%{x}</b><br>비중: %{y:.1f}%'
            )
            
            apply_chart_style(fig2, "🌏 국가별 투자 비중")
            st.plotly_chart(fig2, theme=None, use_container_width=True)
            
        industry_cols = [c for c in df.columns if any(keyword in c.lower() for keyword in ['industry', '업종', 'sector'])]
        if industry_cols:
            industry_col = industry_cols[0]
            industry_df = df.groupby(industry_col)[weight_col].sum().sort_values(ascending=False).reset_index()
            
            # 업종별 바차트
            fig3 = px.bar(
                industry_df,
                x=industry_col,
                y=weight_col,
                title="업종별 비중",
                text=weight_col,
                color=industry_col,
                color_discrete_sequence=COLOR_PALETTES['vibrant']
            )
            
            fig3.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside',
                textfont=dict(size=14, color='black'),
                hovertemplate='<b>%{x}</b><br>비중: %{y:.1f}%'
            )
            
            apply_chart_style(fig3, "🏭 업종별 투자 비중")
            st.plotly_chart(fig3, theme=None, use_container_width=True)
            
            # 업종별 트리맵
            fig_tree = px.treemap(
                df,
                path=[industry_col, ticker_col],
                values=weight_col,
                color=weight_col,
                color_continuous_scale='Viridis',
                hover_data=[weight_col],
                color_continuous_midpoint=df[weight_col].median()
            )
            
            fig_tree.update_traces(
                textfont=dict(size=14, color='white'),
                textposition='middle center',
                hovertemplate='<b>%{label}</b><br>비중: %{value:.1f}%'
            )
            
            fig_tree.update_layout(
                treemapcolorway=px.colors.qualitative.Bold,
                coloraxis_showscale=True,
                margin=dict(t=50, l=25, r=25, b=25)
            )
            
            apply_chart_style(fig_tree, "🌳 업종-종목 투자 트리맵", height=500)
            st.plotly_chart(fig_tree, theme=None, use_container_width=True)
    
    with tabs[1]:
        # 2. 위험-수익 산점도 (새로운 시각화)
        st.subheader("위험-수익 분석")
        
        # 가상의 다른 포트폴리오 데이터 (비교용)
        portfolios = {
            "추천 포트폴리오": {"수익률": 8.5, "위험": 12.3, "샤프비율": 0.69},
            "안정형": {"수익률": 5.2, "위험": 8.1, "샤프비율": 0.64},
            "균형형": {"수익률": 7.4, "위험": 10.8, "샤프비율": 0.68},
            "공격형": {"수익률": 10.1, "위험": 15.6, "샤프비율": 0.65},
            "S&P 500": {"수익률": 9.8, "위험": 16.2, "샤프비율": 0.60},
            "KOSPI": {"수익률": 7.2, "위험": 14.8, "샤프비율": 0.49},
        }
        
        # 데이터프레임 생성
        risk_return_df = pd.DataFrame([
            {"포트폴리오": name, "예상 연간 수익률(%)": data["수익률"], "연간 변동성(%)": data["위험"], "샤프비율": data["샤프비율"]}
            for name, data in portfolios.items()
        ])
        
        # 샤프비율에 따른 버블 크기 설정
        risk_return_df["버블 크기"] = risk_return_df["샤프비율"] * 50
        
        # 위험-수익 산점도
        fig_risk = px.scatter(
            risk_return_df,
            x="연간 변동성(%)",
            y="예상 연간 수익률(%)",
            size="버블 크기",
            color="포트폴리오",
            text="포트폴리오",
            color_discrete_sequence=COLOR_PALETTES['vivid'],
            title="포트폴리오 위험-수익 비교"
        )
        
        fig_risk.update_traces(
            textposition='top center',
            marker=dict(opacity=0.8, line=dict(width=2, color='white')),
            hovertemplate='<b>%{text}</b><br>수익률: %{y:.1f}%<br>변동성: %{x:.1f}%<br>샤프비율: %{customdata[0]:.2f}'
        )
        
        fig_risk.update_traces(customdata=risk_return_df[['샤프비율']])
        
        apply_chart_style(fig_risk, "📊 포트폴리오 효율성 분석")
        st.plotly_chart(fig_risk, theme=None, use_container_width=True)
        
        # 샤프비율 비교 바차트
        fig_sharpe = px.bar(
            risk_return_df.sort_values(by="샤프비율", ascending=False),
            x="포트폴리오",
            y="샤프비율",
            color="포트폴리오",
            color_discrete_sequence=COLOR_PALETTES['dark'],
            title="포트폴리오 샤프비율 비교",
            text="샤프비율"
        )
        
        fig_sharpe.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>샤프비율: %{y:.2f}'
        )
        
        apply_chart_style(fig_sharpe, "📈 위험 조정 수익률 비교")
        st.plotly_chart(fig_sharpe, theme=None, use_container_width=True)
    
    with tabs[2]:
        # 3. 자산 상관관계 히트맵 (새로운 시각화)
        st.subheader("자산 상관관계 분석")
        
        # 샘플 종목 리스트 (상위 8개 종목 또는 전체)
        top_stocks = sorted_df.head(min(8, len(sorted_df)))[ticker_col].tolist()
        
        # 가상의 상관관계 데이터 생성
        np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
        num_stocks = len(top_stocks)
        corr_matrix = np.eye(num_stocks)  # 대각선은 1로 설정 (자기 자신과의 상관관계)
        
        # 상관관계 행렬 채우기 (대칭행렬)
        for i in range(num_stocks):
            for j in range(i+1, num_stocks):
                # -0.3에서 0.9 사이의 상관계수 생성
                corr = np.random.uniform(-0.3, 0.9)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # 대칭 설정
        
        # 데이터프레임으로 변환
        corr_df = pd.DataFrame(corr_matrix, index=top_stocks, columns=top_stocks)
        
        # 히트맵 생성
        fig_corr = px.imshow(
            corr_df,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',  # 빨강(-) 흰색(0) 파랑(+) 스케일
            zmin=-1, zmax=1,  # 상관관계 범위 -1에서 1로 고정
            title="자산 간 상관관계 히트맵"
        )
        
        fig_corr.update_layout(
            height=500,
            xaxis_title="",
            yaxis_title="",
            coloraxis_colorbar=dict(
                title="상관계수",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.0", "-0.5", "0.0", "0.5", "1.0"]
            )
        )
        
        apply_chart_style(fig_corr, "🔄 자산 상관관계 분석")
        st.plotly_chart(fig_corr, theme=None, use_container_width=True)
        
        # 상관관계 해석 설명 추가
        st.markdown("""
        ### 상관관계 해석
        - **양의 상관관계(0~1)**: 두 자산의 가격이 같은 방향으로 움직이는 경향이 있습니다.
        - **음의 상관관계(-1~0)**: 두 자산의 가격이 반대 방향으로 움직이는 경향이 있습니다.
        - **낮은 상관관계**: 서로 다른 움직임을 보이는 자산들은 분산투자 효과를 높여줍니다.
        """)
    
    with tabs[3]:
        # 4. 미래 수익률 시뮬레이션 (몬테카를로) (새로운 시각화)
        st.subheader("포트폴리오 미래 수익률 시뮬레이션")
        
        # 시뮬레이션 설정
        initial_investment = 10000  # 초기 투자금 ($10,000)
        years = 10  # 10년 시뮬레이션
        simulations = 500  # 시뮬레이션 횟수
        
        # 포트폴리오 연간 수익률 및 변동성 (가정)
        annual_return = 0.08  # 8% 기대 수익률
        annual_volatility = 0.12  # 12% 표준편차
        
        # 시간 경과에 따른 투자 가치 시뮬레이션
        np.random.seed(42)
        simulation_df = pd.DataFrame()
        
        for i in range(simulations):
            # 각 연도별 수익률 시뮬레이션
            returns = np.random.normal(annual_return, annual_volatility, years)
            # 복리 수익 계산
            values = [initial_investment]
            for r in returns:
                values.append(values[-1] * (1 + r))
            
            simulation_df[f'sim_{i}'] = values
        
        # 시간 축 생성
        simulation_df['year'] = range(years + 1)
        
        # 중앙값, 상위 10%, 하위 10% 계산
        median_values = simulation_df.drop('year', axis=1).median(axis=1)
        upper_10 = simulation_df.drop('year', axis=1).quantile(0.9, axis=1)
        lower_10 = simulation_df.drop('year', axis=1).quantile(0.1, axis=1)
        
        # 100개의 랜덤 시뮬레이션 선택 (모든 선 표시하면 너무 복잡함)
        random_sims = np.random.choice(simulations, 100, replace=False)
        
        # 시뮬레이션 플롯 생성
        fig_sim = go.Figure()
        
        # 랜덤 시뮬레이션 경로 추가
        for i in random_sims:
            fig_sim.add_trace(
                go.Scatter(
                    x=simulation_df['year'],
                    y=simulation_df[f'sim_{i}'],
                    mode='lines',
                    line=dict(color='rgba(200, 200, 200, 0.2)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # 중앙값, 상위 10%, 하위 10% 추가
        fig_sim.add_trace(
            go.Scatter(
                x=simulation_df['year'],
                y=median_values,
                mode='lines',
                line=dict(color='blue', width=3),
                name='중앙값',
                hovertemplate='연도: %{x}<br>가치: $%{y:.0f}'
            )
        )
        
        fig_sim.add_trace(
            go.Scatter(
                x=simulation_df['year'],
                y=upper_10,
                mode='lines',
                line=dict(color='green', width=2),
                name='상위 10%',
                hovertemplate='연도: %{x}<br>가치: $%{y:.0f}'
            )
        )
        
        fig_sim.add_trace(
            go.Scatter(
                x=simulation_df['year'],
                y=lower_10,
                mode='lines',
                line=dict(color='red', width=2),
                name='하위 10%',
                hovertemplate='연도: %{x}<br>가치: $%{y:.0f}'
            )
        )
        
        fig_sim.update_layout(
            title='10년 포트폴리오 가치 시뮬레이션',
            xaxis_title='연도',
            yaxis_title='포트폴리오 가치 ($)',
            yaxis_tickprefix='$',
            yaxis_tickformat=',',
            hovermode='x unified'
        )
        
        apply_chart_style(fig_sim, "🔮 포트폴리오 미래 가치 시뮬레이션", height=500)
        st.plotly_chart(fig_sim, theme=None, use_container_width=True)
        
        # 최종 투자 결과 분포
        final_values = simulation_df.iloc[-1].drop('year')
        
        # 히스토그램
        fig_hist = px.histogram(
            final_values,
            nbins=30,
            title="10년 후 투자 결과 분포",
            color_discrete_sequence=['rgba(0, 128, 255, 0.7)']
        )
        
        fig_hist.add_vline(
            x=median_values.iloc[-1],
            line_dash="dash",
            line_color="blue",
            annotation_text=f"중앙값: ${median_values.iloc[-1]:.0f}",
            annotation_position="top right"
        )
        
        fig_hist.update_layout(
            xaxis_title="포트폴리오 가치 ($)",
            yaxis_title="시뮬레이션 횟수",
            xaxis_tickprefix='$',
            xaxis_tickformat=',',
            showlegend=False
        )
        
        apply_chart_style(fig_hist, "📊 10년 후 포트폴리오 가치 분포")
        st.plotly_chart(fig_hist, theme=None, use_container_width=True)
    
    with tabs[4]:
        # 5. 지역별 분포 지도 (새로운 시각화)
        st.subheader("글로벌 투자 분포")
        
        # 국가별 비중 데이터 (국가 정보가 있는 경우)
        if 'country_col' in locals() and country_col:
            country_data = df.groupby(country_col)[weight_col].sum().reset_index()
            
            # 국가 영문명으로 변환 (한글 국가명인 경우)
            country_mapping = {
                "한국": "South Korea", "미국": "United States", "중국": "China", "일본": "Japan",
                "영국": "United Kingdom", "독일": "Germany", "프랑스": "France", "인도": "India",
                "브라질": "Brazil", "캐나다": "Canada", "호주": "Australia", "러시아": "Russia"
            }
            
            country_data[country_col] = country_data[country_col].map(
                lambda x: country_mapping.get(x, x)
            )
            
            # 지도 시각화
            fig_map = px.choropleth(
                country_data,
                locations=country_col,
                locationmode="country names",
                color=weight_col,
                hover_name=country_col,
                color_continuous_scale="Viridis",
                title="글로벌 지역별 투자 비중",
                projection="natural earth"
            )
            
            fig_map.update_layout(
                height=550,
                coloraxis_colorbar=dict(title="투자 비중 (%)")
            )
            
            apply_chart_style(fig_map, "🌎 글로벌 투자 분포")
            st.plotly_chart(fig_map, theme=None, use_container_width=True)
        else:
            # 국가 정보가 없는 경우 가상 데이터로 지도 표시
            st.warning("포트폴리오에 국가 정보가 포함되어 있지 않아 샘플 데이터로 지도를 표시합니다.")
            
            sample_countries = {
                "United States": 45, "South Korea": 20, "China": 10, "Japan": 5,
                "Germany": 5, "United Kingdom": 5, "India": 5, "Brazil": 5
            }
            
            country_data = pd.DataFrame([
                {"country": country, "weight": weight}
                for country, weight in sample_countries.items()
            ])
            
            fig_map = px.choropleth(
                country_data,
                locations="country",
                locationmode="country names",
                color="weight",
                hover_name="country",
                color_continuous_scale="Viridis",
                title="샘플 글로벌 투자 분포",
                projection="natural earth"
            )
            
            fig_map.update_layout(
                height=550,
                coloraxis_colorbar=dict(title="투자 비중 (%)")
            )
            
            apply_chart_style(fig_map, "🌎 샘플 글로벌 투자 분포")
            st.plotly_chart(fig_map, theme=None, use_container_width=True)

def render_agentic_rag_tab():
    st.header("맞춤형 포트폴리오 추천")
    survey = portfolio_survey()

    # 문서 자동 로드
    vectorstore = load_predefined_documents()

    # LLM 초기화
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.3)
    llm_gpt4 = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.3)

    # Tool 정의 (영문명 필수)
    tools = [
        Tool(
            name="portfolio_analysis",
            func=lambda x: analyze_portfolio(survey, llm_gpt4),
            description="Analyze the user's stock survey and recommend an optimal portfolio."
        ),
        Tool(
            name="stock_chart",
            func=lambda ticker: plot_advanced_stock_chart(ticker),
            description="Get stock chart and info by ticker symbol."
        )
    ]
    if vectorstore:
        tools.append(
            Tool(
                name="market_report_search",
                func=lambda q: rag_search(q, vectorstore, llm),
                description="Search uploaded market analysis documents."
            )
        )

    # Agent 초기화
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=False, memory=memory
    )

    # 포트폴리오 추천 버튼 & 결과 시각화
    st.markdown("#### 📊 내게 맞는 포트폴리오 추천")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("포트폴리오 추천받기", type="primary", use_container_width=True):
            with st.spinner("AI가 개인화 포트폴리오를 추천 중..."):
                try:
                    # 포트폴리오 분석 함수 직접 호출 (속도 개선)
                    answer = analyze_portfolio(survey, llm_gpt4)
                    table_md, explanation = extract_markdown_table(answer)
                    
                    st.success("✅ 추천 포트폴리오가 생성되었습니다.")
                    
                    if explanation:
                        with st.expander("📝 추천 포트폴리오 설명", expanded=True):
                            st.markdown(explanation)
                    
                    if table_md:
                        df = parse_portfolio_table(table_md)
                        if df is not None:
                            plot_portfolio_interactive(df)
                        else:
                            st.error("포트폴리오 데이터 파싱에 실패했습니다.")
                except Exception as e:
                    st.error(f"추천 오류: {e}")

    # 미리 정의된 프롬프트 버튼 (예쁜 UI)
    st.markdown("#### 📝 전문가 투자 분석 요청")
    
    # 버튼 그리드 (2열 레이아웃)
    col1, col2 = st.columns(2)
    
    for i, (button_text, prompt) in enumerate(INVESTMENT_PROMPTS.items()):
        col = col1 if i % 2 == 0 else col2
        if col.button(button_text, key=f"btn_{i}", use_container_width=True):
            with st.spinner(f"{button_text.strip('💹📈💰🛡️🌎📊🏢')} 분석 중..."):
                try:
                    # 티커 추출 (있는 경우 차트 먼저 표시)
                    ticker_search = re.search(r'\((.*?)\)', button_text)
                    if ticker_search:
                        ticker = ticker_search.group(1)
                        plot_advanced_stock_chart(ticker)
                    
                    response = agent.run(prompt)
                    st.markdown(f"### {button_text.strip('💹📈💰🛡️🌎📊🏢')} 결과")
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"분석 오류: {e}")
