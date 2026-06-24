import os
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from datetime import datetime, timedelta
import requests

# ==============================================================================
# [설정 항목] 드라이브 경로 및 자격 증명 KEY 매핑
# ==============================================================================
BASE_VAULT_PATH = r"D:\ETF-Hunter"
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    genai.configure(api_key=GEMINI_API_KEY)

# 설계도 기반 폴더 트리 자동 보장
folders = ["01_Daily", "02_Weekly", "03_Monthly", "04_ETF", "05_Macro", "06_SmartMoney", "07_Journal", "08_Backtest"]
for f in folders:
    os.makedirs(os.path.join(BASE_VAULT_PATH, f), exist_ok=True)

etfs = ["AIQ", "BOTZ", "SMH", "SOXX", "PAVE"]
macros = {"DXY": "DX-Y.NYB", "VIX": "^VIX", "USDJPY": "JPY=X"}
smart_money = {"Yen": "JPY=X", "Gold": "GC=F", "Treasury": "^TNX"}

print("==============================================================================")
print("  [ETF HUNTER] Gemini 자체 페르소나 상호 반론 및 ETF Dataview 마스터 가동")
print("==============================================================================")

# ==============================================================================
# 1. 데이터 수집 및 기술적 분석 / 신규 엔진 지표 계산
# ==============================================================================
today_str = datetime.today().strftime('%Y-%m-%d')
all_tickers = etfs + list(macros.values()) + list(smart_money.values())
raw = yf.download(all_tickers, period="1y", auto_adjust=True)
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
# 2. Gemini 내 복수 페르소나 상호 검증 및 8대 출력 항목 매핑
# ==============================================================================
ai_decision = "AI 분석 미이행"
permission_grade = "B"
total_score = 80

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    try:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            model.generate_content("test")
        except:
            model = genai.GenerativeModel('gemini-1.5-flash')
            
        data_summary = f"시장국면 변수: Regime={market_regime}, Fear Score={fear_score:.2f}, Carry Risk={carry_risk:.2f}\n매크로 데이터: DXY={close_data['DX-Y.NYB'].iloc[-1]:.2f}, VIX={vix_latest:.2f}, USDJPY={usdjpy_series.iloc[-1]:.2f}\nETF 퀀트 랭킹판:\n{df_etf.to_string()}"
        
        # 2-1. 주 분석가의 매수 관점 의견
        print("▶ [AI 검증 1단계] 주 분석가 페르소나 의견 수집 중...")
        p1 = f"당신은 헤지펀드 주 분석가입니다. 데이터를 보고 오늘 진입할 원탑 ETF인 {top_etf}의 매수 강점 위주로 의견을 작성하세요.\n{data_summary}"
        ans1 = model.generate_content(p1).text
        
        # 2-2. 리스크 위험관리관의 반론 관점 의견 (Grok 역할 대리)
        print("▶ [AI 검증 2단계] 리스크 관리관 페르소나 반론 수집 중...")
        p2 = f"당신은 철저히 위험을 경고하는 까칠한 리스크 매니저입니다. 주 분석가의 다음 추천 의견과 매크로 지표의 맹점을 찔러 강하게 반론하세요.\n[추천의견]:\n{ans1}\n[데이터]:\n{data_summary}"
        ans2 = model.generate_content(p2).text
        
        # 2-3. 최종 포트폴리오 매니저(CIO)의 결론 조율 (설계도 7단계 스펙)
        print("▶ [AI 검증 3단계] 수석 CIO 엔진 최종 종합 의사결정 중...")
        p3 = f"당신은 글로벌 매크로 헤지펀드 수석 CIO입니다. 주 분석가의 낙관론과 리스크 관리관의 비관론을 종합하여 최종 결론 리포트를 마크다운으로 출력하십시오.\n\n[주 분석가]:\n{ans1}\n\n[리스크 관리관 반론]:\n{ans2}\n\n[필수 출력 서식 규격]\n### 1 시장국면\n[내용 작성]\n### 2 스마트머니 흐름\n[내용 작성]\n### 3 ETF 랭킹\n[내용 작성]\n### 4 매수허가등급\n- **추천 티커**: {top_etf}\n- **허가 등급**: [A / B / C 중 최종 선택]\n- **종합 점수**: [90점 만점 기준 최종 환산 점수 숫자만]\n### 5 리스크\n[내용 작성]\n### 6 단기전망\n[내용 작성]\n### 7 중기전망\n[내용 작성]\n### 8 장기전망\n[내용 작성]"
        response = model.generate_content(prompt=p3)
        ai_decision = response.text
        
        if "허가 등급" in ai_decision:
            try:
                permission_grade = ai_decision.split("허가 등급**:")[1].split("\n")[0].strip()[:1]
                total_score = int(''.join(filter(str.isdigit, ai_decision.split("종합 점수**:")[1].split("\n")[0])))
            except:
                pass
    except Exception as e:
        ai_decision = f"검증 엔진 연산 오류: {str(e)}"

# ==============================================================================
# 3. 설계도 6단계: Backtest 시계열 성과 역산 추적 엔진
# ==============================================================================
print("▶ [BACKTEST] 7일 / 30일 시차 성과 추적 데이터 가공 중...")
bt_file = os.path.join(BASE_VAULT_PATH, "08_Backtest", "backtest_history.csv")
if os.path.exists(bt_file):
    df_bt = pd.read_csv(bt_file)
else:
    df_bt = pd.DataFrame(columns=["Date", "Ticker", "Entry_Price", "Price_7d", "Price_30d", "Return_7d(%)", "Return_30d(%)"])

today_dt = datetime.today()
for idx, row in df_bt.iterrows():
    row_date = datetime.strptime(row["Date"], "%Y-%m-%d")
    tk = row["Ticker"]
    if pd.isna(row["Price_7d"]) and (today_dt - row_date).days >= 7:
        try:
            p_7d = close_data[tk].loc[row["Date"]:]
            if len(p_7d) >= 5:
                actual_p = p_7d.iloc[5]
                df_bt.at[idx, "Price_7d"] = round(actual_p, 2)
                df_bt.at[idx, "Return_7d(%)"] = round(((actual_p - row["Entry_Price"]) / row["Entry_Price"]) * 100, 2)
        except:
            pass
    if pd.isna(row["Price_30d"]) and (today_dt - row_date).days >= 30:
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

# ==============================================================================
# 4. 설계도 9단계: 옵시디언 Dataview 최적화 데이터 적재
# ==============================================================================
print("▶ [DB 저장] Dataview 포맷(YAML Frontmatter) 기반 파일 저장 중...")

# 가. 01_Daily 보고서 생성
daily_meta = f"---\ntype: daily\ndate: {today_str}\nregime: {market_regime}\nfear: {fear_score:.2f}\ncarry_risk: {carry_risk:.2f}\ntop_pick: {top_etf}\npermission: {permission_grade}\nscore: {total_score}\n---\n# {today_str} 데일리 리포트\n\n## 📊 퀀트 스크리닝 순위\n"
for idx, row in df_etf.iterrows():
    daily_meta += f"- {idx}위: [[{row['Ticker']}]] (종가: ${row['Close']} | RS: {row['RS']} | 압축: {row['Compression']})\n"

with open(os.path.join(BASE_VAULT_PATH, "01_Daily", f"{today_str}.md"), "w", encoding="utf-8") as f:
    f.write(daily_meta)

# 나. 04_ETF 개별 종목별 검증용 YAML 데이터 동적 누적 개편
for idx, row in df_etf.iterrows():
    f_path = os.path.join(BASE_VAULT_PATH, "04_ETF", f"{row['Ticker']}.md")
    
    new_yaml_block = f"---" + f"\nticker: {row['Ticker']}\nscore: {total_score if row['Ticker'] == top_etf else 0}\nfear: {int(fear_score)}\npermission: {permission_grade if row['Ticker'] == top_etf else 'N/A'}\ndate: {today_str}\nprice: {row['Close']}\n---\n"
    existing_content = ""
    if os.path.exists(f_path):
        with open(f_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
            if existing_content.startswith("---"):
                parts = existing_content.split("---", 2)
                if len(parts) >= 3:
                    existing_content = parts[2].strip()

    if not existing_content:
        existing_content = f"\n# {row['Ticker']} 리서치 DB\n\n## 📈 누적 기술적 지표\n| 날짜 | 종가 | RS점수 | 압축점수 |\n| :--- | :--- | :--- | :--- |\n"
    
    updated_content = new_yaml_block + existing_content + f"| {today_str} | ${row['Close']} | {row['RS']} | {row['Compression']} |\n"
    with open(f_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

# 다. 07_Journal 투자일지 생성
journal_meta = f"---\ntype: journal\ndate: {today_str}\ntop_pick: {top_etf}\n---\n# 🧠 매크로 CIO 투자판단 저널 ({today_str})\n\n{ai_decision}\n"
with open(os.path.join(BASE_VAULT_PATH, "07_Journal", f"{today_str}.md"), "w", encoding="utf-8") as f:
    f.write(journal_meta)

# 라. 08_Backtest 대시보드 업데이트
bt_markdown = f"# 📉 퀀트 타임프레임 성과 추적기\n*업데이트: {today_str}*\n\n| 추천일 | 자산군 | 진입가 | 7일후 성과 | 30일후 성과 |\n| :---: | :---: | :---: | :---: | :---: |\n"
for _, row in df_bt.tail(10).iloc[::-1].iterrows():
    ret_7 = f"{row['Return_7d(%)']}%" if not pd.isna(row['Return_7d(%)']) else "⏳ 대기중"
    ret_30 = f"{row['Return_30d(%)']}%" if not pd.isna(row['Return_30d(%)']) else "⏳ 대기중"
    bt_markdown += f"| {row['Date']} | **[[{row['Ticker']}]]** | ${row['Entry_Price']} | {ret_7} | {ret_30} |\n"

with open(os.path.join(BASE_VAULT_PATH, "08_Backtest", "Performance_Dashboard.md"), "w", encoding="utf-8") as f:
    f.write(bt_markdown)

# ==============================================================================
# 5. Telegram 메시지 최종 전송
# ==============================================================================
if TELEGRAM_TOKEN and TELEGRAM_TOKEN != "YOUR_TELEGRAM_BOT_TOKEN":
    tg_message = f"🔔 [ETF-HUNTER Dual-Gemini] {today_str} 시그널\n\n📊 [시장 국면]\n- Regime: {market_regime}\n- Fear: {fear_score:.1f} / Carry: {carry_risk:.1f}\n- 주도 자산: {top_etf} (등급: {permission_grade} / 점수: {total_score}점)\n\n⚖️ [Gemini 페르소나 상호 교차검증 리포트]\n{ai_decision[:650]}..."
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": tg_message})
        print("[성공] 텔레그램 송신 완료")
    except Exception as e:
        print(f"텔레그램 전송 실패: {e}")

print("==============================================================================")
print("  [완료] Gemini 상호 검증 ➡️ Dataview 파싱 규격 저장 ➡️ 텔레그램 송신 완료")
print("==============================================================================")
