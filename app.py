import streamlit as st  # 가장 먼저 streamlit 임포트

# 반드시 다른 모든 import와 코드보다 먼저 set_page_config 호출
st.set_page_config(
    page_title="🤖 AI 금융 어시스턴트",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 그 다음에 모든 다른 임포트
import os
from dotenv import load_dotenv
from pages.stock_search import render_stock_search
from pages.document_search import render_document_search
from pages.agentic_rag import render_agentic_rag_tab

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 사이드바: 모델 & 하이퍼파라미터
with st.sidebar:
    st.header("⚙️ 설정")
    mv = st.radio("모델 선택", ["GPT-3.5 Turbo","GPT-4","GEMINI"], index=0, key="model_version")
    if mv=="GEMINI":
        st.selectbox("Gemini 모델", ["gemini-1.5-flash","gemini-2.0-flash"], key="gemini_model")
    st.markdown("---")
    st.subheader("API 하이퍼파라미터")
    st.slider("Temperature", 0.0,1.0,0.7,0.1, key="temperature")
    st.slider("Top P",       0.0,1.0,0.9,0.1, key="top_p")
    st.slider("Max Tokens",100,4000,1000,100, key="max_tokens")
    st.slider("Freq Penalty",-2.0,2.0,0.0,0.1, key="frequency_penalty")
    st.slider("Pres Penalty",-2.0,2.0,0.0,0.1, key="presence_penalty")
    
    # haha what's goin on

st.title("🤖 AI 금융 어시스턴트")
tab1, tab2, tab3 = st.tabs(["📈 주식 정보","📄 문서 분석", "📊 포트폴리오 추천"])
with tab1: render_stock_search()
with tab2: render_document_search()
with tab3: render_agentic_rag_tab()
