import os
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime
import requests

# ==============================================================================
# [설정 항목]
# ==============================================================================
BASE_VAULT_PATH = os.getenv("BASE_VAULT_PATH", os.getcwd())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

folders = ["Daily", "Weekly", "Monthly", "ETF", "Macro", "SmartMoney", "Journal", "Backtest"]
for f in folders:
    os.makedirs(os.path.join(BASE_VAULT_PATH, f), exist_ok=True)

etfs = ["AIQ", "BOTZ", "SMH", "SOXX", "PAVE"]
macros = {"DXY": "DX-Y.NYB", "VIX": "^VIX", "USDJPY": "JPY=X"}
smart_money = {"Yen": "JPY=X", "Gold": "GC=F", "Treasury": "^TNX"}

print("==============================================================================")
print(" [ETF HUNTER] Gemini 자체 페르소나 상호 반론 및 ETF Dataview 마스터 가동")
print("==============================================================================")

# ==============================================================================
# 1. 데이터 수집 및 기술적 분석
# ==============================================================================
today_str = datetime.today().strftime('%Y-%m-%d')
all_tickers = etfs + list(macros.values()) + list(smart_money.values())

raw = yfinance.download(all_tickers, period="1y", auto_adjust=True)
close_data = raw["Close"].ffill()
volume_data = raw["Volume"].ffill() if "Volume" in raw else pd.DataFrame(index=raw.index, columns=all_tickers).fillna(1)

vix_latest = close_data["^VIX"].iloc[-1]
vix_mean = close_data["^VIX"].rolling(20).mean().iloc[-1]
fear_score = min(max((vix_latest / vix_mean) * 50, 0), 100)

usdjpy_series = close_data["JPY=X"]
usdjpy_change = (usdjpy_series.iloc[-1] - usdjpy_series.iloc[-10]) / usdjpy_series.iloc[-10] * 100
carry_risk = min(max(-usdjpy_change * 20 + 30, 0), 100)

market_regime = "Risk On" if fear_score < 60 and carry_risk < 50 else "Risk Off"

etf_results = []
for ticker in etfs:
    price = close_data[ticker]
    vol = volume_data[ticker]
   
    ema20 = price.ewm(span=20, adjust=False).mean().iloc[-1]
    vwap = ((price * vol).rolling(14).sum() / vol.rolling(14).sum()).iloc[-1]
    if np.isnan(vwap):
        vwap = price.iloc[-1]
    
    exp1 = price.ewm(span=12, adjust=False).mean()
    exp2 = price.ewm(span=26, adjust=False).mean()
    macd = (exp1 - exp2).iloc[-1]
   
    ret_3m = (price.iloc[-1] - price.iloc[-60]) / price.iloc[-60]
    ret_1m = (price.iloc[-1] - price.iloc[-20]) / price.iloc[-20]
    rs_score = (ret_3m * 0.7 + ret_1m * 0.3) * 100
   
    ema_spans = [2, 4, 8, 16, 32, 64, 128]
    ema_vals = [price.ewm(span=s, adjust=False).mean().iloc[-1] for s in ema_spans]
    compression_score = 100 / (1 + (np.std(ema_vals) / price.iloc[-1] * 10))
   
    etf_results.append({
        "Ticker": ticker, "Close": round(price.iloc[-1], 2),
        "RS": round(rs_score, 2), "Compression": round(compression_score, 2),
        "EMA20": round(ema20, 2), "VWAP": round(vwap, 2), "MACD": round(macd, 2)
    })

df_etf = pd.DataFrame(etf_results).sort_values(by="RS", ascending=False).reset_index(drop=True)
df_etf.index = df_etf.index + 1
top_etf = df_etf.iloc[0]['Ticker']

# ==============================================================================
# 2. Gemini 분석
# ==============================================================================
ai_decision = "AI 분석 미이행 (API 키 없음)"
permission_grade = "B"
total_score = 80

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    GEMINI_API_KEY = GEMINI_API_KEY.strip()
    print(f"✅ Gemini API Key 감지됨 (길이: {len(GEMINI_API_KEY)})")
    genai.configure(api_key=GEMINI_API_KEY)

    model = None
    for model_name in ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.5-flash-lite']:
        try:
            model = genai.GenerativeModel(model_name)
            test_response = model.generate_content("Say hello in one word.")
            print(f"✅ 모델 로드 성공: {model_name}")
            break
        except Exception as e:
            print(f"⚠️ {model_name} 실패: {str(e)[:100]}...")
            continue

    if model is None:
        print("❌ 모든 Gemini 모델 로드 실패")
        ai_decision = "Gemini 모델 초기화 실패"
    else:
        try:
            data_summary = f"""
- 시장국면 변수: Regime={market_regime}, Fear Score={fear_score:.2f}, Carry Risk={carry_risk:.2f}
- 매크로 데이터: DXY={close_data.get('DX-Y.NYB', pd.Series([0])).iloc[-1]:.2f}, VIX={vix_latest:.2f}, USDJPY={usdjpy_series.iloc[-1]:.2f}
- ETF 퀀트 랭킹판:
{df_etf.to_string()}
"""

            print("▶ [AI 검증 1단계] 주 분석가 의견 수집 중...")
            p1 = f"당신은 헤지펀드 주 분석가입니다. 데이터를 보고 오늘 진입할 원탑 ETF인 {top_etf}의 매수 강점 위주로 의견을 작성하세요.\n{data_summary}"
            ans1 = model.generate_content(p1).text

            print("▶ [AI 검증 2단계] 리스크 관리관 반론 수집 중...")
            p2 = f"당신은 리스크 매니저입니다. 주 분석가의 다음 추천 의견과 매크로 지표의 맹점을 찔러 강하게 반론하세요.\n[추천의견]:\n{ans1}\n[데이터]:\n{data_summary}"
            ans2 = model.generate_content(p2).text

            print("▶ [AI 검증 3단계] 수석 CIO 엔진 최종 종합 의사결정 중...")
            p3 = f"""당신은 글로벌 매크로 헤지펀드 수석 CIO입니다. 주 분석가의 낙관론과 리스크 관리관의 비관론을 종합하여 최종 결론 리포트를 마크다운으로 출력하십시오.

[주 분석가]:
{ans1}

[리스크 관리관 반론]:
{ans2}

[필수 출력 서식 규격]
### 1 시장국면
### 2 스마트머니 흐름
### 3 ETF 랭킹
### 4 매수허가등급
- **추천 티커**: {top_etf}
- **허가 등급**: [A / B / C 선택]
- **종합 점수**: [90점 만점 기준 숫자만]
### 5 리스크
### 6 단기전망
### 7 중기전망
### 8 장기전망
"""
            response = model.generate_content(p3)
            ai_decision = response.text

            if "허가 등급" in ai_decision:
                try:
                    permission_grade = ai_decision.split("허가 등급**:")[1].split("\n")[0].strip()[:1]
                    total_score = int(''.join(filter(str.isdigit, ai_decision.split("종합 점수**:")[1].split("\n")[0])))
                except:
                    pass
        except Exception as e:
            print(f"❌ Gemini 분석 중 오류: {str(e)}")
            ai_decision = f"검증 엔진 연산 오류: {str(e)}"
else:
    print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다.")

# ==============================================================================
# 3. 백테스트 & 파일 저장 (나머지 부분 동일)
# ==============================================================================
print("▶ [BACKTEST] 7일 / 30일 시차 성과 추적 데이터 가공 중...")
bt_file = os.path.join(BASE_VAULT_PATH, "Backtest", "backtest_history.csv")

if os.path.exists(bt_file):
    df_bt = pd.read_csv(bt_file)
else:
    df_bt = pd.DataFrame(columns=["Date", "Ticker", "Entry_Price", "Price_7d", "Price_30d", "Return_7d(%)", "Return_30d(%)"])

today_dt = datetime.today()
for idx, row in df_bt.iterrows():
    row_date = datetime.strptime(row["Date"], "%Y-%m-%d")
    tk = row["Ticker"]
    if pd.isna(row.get("Price_7d")) and (today_dt - row_date).days >= 7:
        try:
            p_7d = close_data[tk].loc[row["Date"]:]
            if len(p_7d) >= 5:
                actual_p = p_7d.iloc[5]
                df_bt.at[idx, "Price_7d"] = round(actual_p, 2)
                df_bt.at[idx, "Return_7d(%)"] = round(((actual_p - row["Entry_Price"]) / row["Entry_Price"]) * 100, 2)
        except:
            pass
    if pd.isna(row.get("Price_30d")) and (today_dt - row_date).days >= 30:
        try:
            p_30d = close_data[tk].loc[row["Date"]:]
            if len(p_30d) >= 20:
                actual_p = p_30d.iloc[20]
                df_bt.at[idx, "Price_30d"] = round(actual_p, 2)
                df_bt.at[idx, "Return_30d(%)"] = round(((actual_p - row["Entry_Price"]) / row["Entry_Price"]) * 100, 2)
        except:
            pass

today_entry_price = round(close_data[top_etf].iloc[-1], 2)
new_bt_row = pd.DataFrame([{"Date": today_str, "Ticker": top_etf, "Entry_Price": today_entry_price, "Price_7d": np.nan, "Price_30d": np.nan, "Return_7d(%)": np.nan, "Return_30d(%)": np.nan}])
df_bt = pd.concat([df_bt, new_bt_row]).drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
df_bt.to_csv(bt_file, index=False)

# Daily, ETF, Journal, Backtest 저장 부분은 이전 코드와 동일하게 유지 (생략했으나 필요시 이전 버전 참조)

print("==============================================================================")
print(" [완료] Gemini 상호 검증 ➡️ Dataview 파싱 규격 저장 완료")
print("==============================================================================")
