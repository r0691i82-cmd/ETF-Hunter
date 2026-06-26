import os
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import requests
import json

# ==============================================================================
# [설정 및 초기화]
# ==============================================================================
BASE_VAULT_PATH = os.getenv("BASE_VAULT_PATH", os.getcwd())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

folders = ["Daily", "ETF", "Journal", "Backtest"]
for f in folders:
    os.makedirs(os.path.join(BASE_VAULT_PATH, f), exist_ok=True)

# ==============================================================================
# 1. 데이터 수집 및 엔진 구동
# ==============================================================================
def get_market_data():
    etfs = ["AIQ", "BOTZ", "SMH", "SOXX", "PAVE"]
    macros = {"DXY": "DX-Y.NYB", "VIX": "^VIX", "USDJPY": "JPY=X", "TNX": "^TNX"}
    all_tickers = etfs + list(macros.values())
    
    raw = yf.download(all_tickers, period="3mo", auto_adjust=True)
    close = raw["Close"].ffill()
    
    # 엔진 계산
    data = {}
    for t in etfs:
        p = close[t]
        ema20 = p.ewm(span=20, adjust=False).mean().iloc[-1]
        rs = ((p.iloc[-1] - p.iloc[-60]) / p.iloc[-60] * 0.7 + (p.iloc[-1] - p.iloc[-20]) / p.iloc[-20] * 0.3) * 100
        data[t] = {"Close": round(p.iloc[-1], 2), "RS": round(rs, 2), "EMA20": round(ema20, 2)}
        
    return data, close

market_data, close_df = get_market_data()
df_etf = pd.DataFrame(market_data).T.sort_values(by="RS", ascending=False)
top_etf = df_etf.index[0]

# ==============================================================================
# 2. Gemini 통합 분석
# ==============================================================================
def run_cio_engine(market_data, close_df):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    당신은 글로벌 매크로 헤지펀드 CIO입니다. 다음 데이터를 바탕으로 전문적인 리포트를 마크다운 형식으로 작성하세요.
    
    [데이터]
    - 시장 상태: {market_data}
    - 매크로 지표: DXY={close_df['DX-Y.NYB'].iloc[-1]:.2f}, VIX={close_df['^VIX'].iloc[-1]:.2f}
    
    [출력 요구사항]
    1. 시장국면 (Risk On/Off 판정)
    2. ETF 랭킹 및 상세 기술적 분석
    3. 추천 티커 및 매수허가등급 (A/B/C)
    4. 오늘 행동 및 리스크 요인
    """
    response = model.generate_content(prompt)
    return response.text

ai_report = run_cio_engine(market_data, close_df)

# ==============================================================================
# 3. 통합 파일 저장 (Daily)
# ==============================================================================
today_str = datetime.today().strftime('%Y-%m-%d')
file_path = os.path.join(BASE_VAULT_PATH, "Daily", f"{today_str}.md")

final_content = f"""---
date: {today_str}
top_pick: {top_etf}
---
# {today_str} 데일리 통합 리포트

## 🧠 CIO 종합 분석
{ai_report}

## 📊 기술적 퀀트 데이터
{df_etf.to_markdown()}
"""

with open(file_path, "w", encoding="utf-8") as f:
    f.write(final_content)

print(f"✅ 통합 리포트 저장 완료: {file_path}")

# ==============================================================================
# 4. Telegram 전송
# ==============================================================================
if TELEGRAM_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🔔 {today_str} 리포트 생성 완료. 옵시디언을 확인하세요."})
