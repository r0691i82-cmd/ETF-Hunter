import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import google.generativeai as genai

# [설정] 경로와 API 키
BASE_VAULT_PATH = os.getenv("BASE_VAULT_PATH", os.getcwd())
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")

def get_data():
    # 데이터 수집 (매크로 + ETF)
    tickers = ["AIQ", "BOTZ", "PAVE", "SMH", "SOXX", "DX-Y.NYB", "^VIX", "JPY=X", "^TNX"]
    raw = yf.download(tickers, period="3mo", auto_adjust=True)["Close"].ffill()
    
    # 지표 계산
    res = {}
    for t in ["AIQ", "BOTZ", "PAVE", "SMH", "SOXX"]:
        p = raw[t]
        res[t] = {"종가": round(p.iloc[-1], 2), "RS": round(((p.iloc[-1]-p.iloc[-60])/p.iloc[-60])*100, 2)}
    
    macro = {
        "DXY": round(raw["DX-Y.NYB"].iloc[-1], 2),
        "VIX": round(raw["^VIX"].iloc[-1], 2),
        "USDJPY": round(raw["JPY=X"].iloc[-1], 2),
        "10Y_Yield": round(raw["^TNX"].iloc[-1], 2)
    }
    return pd.DataFrame(res).T.sort_values("RS", ascending=False), macro

def get_ai_analysis(df, macro):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY": return "⚠️ API Key 미설정으로 AI 분석 불가"
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"매크로 데이터({macro})와 ETF 랭킹({df.to_string()})을 분석하여 시장 국면, 스마트머니 흐름, 투자 전략을 마크다운으로 상세히 작성해줘."
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"⚠️ AI 분석 일시 오류: {str(e)}"

# 실행 로직
df, macro = get_data()
ai_text = get_ai_analysis(df, macro)
today = datetime.today().strftime('%Y-%m-%d')

# 파일 저장 (01_Daily 폴더에 통합)
content = f"""---
date: {today}
type: daily
---
# {today} 통합 매크로 리포트

## 🌐 시장 환경 및 매크로
{pd.DataFrame([macro]).to_markdown(index=False)}

## 🧠 CIO 종합 전략 (AI)
{ai_text}

## 📊 ETF 퀀트 랭킹
{df.to_markdown()}
"""

save_path = os.path.join(BASE_VAULT_PATH, "01_Daily", f"{today}.md")
with open(save_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"✅ 리포트 생성 완료: {save_path}")
