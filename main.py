from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database import init_db, get_db, ActivityLog
from pydantic import BaseModel
import httpx
import os
import xml.etree.ElementTree as ET
import os
from dotenv import load_dotenv

# Explicitly load .env from the backend directory with absolute path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
try:
    load_dotenv(dotenv_path=env_path, override=True)
except Exception:
    pass

# Foolproof fallback: manually parse .env if python-dotenv fails in uvicorn
if not os.getenv("GEMINI_API_KEY") and os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GEMINI_API_KEY="):
                os.environ["GEMINI_API_KEY"] = line.split("=", 1)[1].strip()

app = FastAPI(title="EBM Appraisal Tool API", version="1.0.0")

@app.on_event("startup")
def on_startup():
    init_db()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PicoRequest(BaseModel):
    p: str
    i: str
    c: str
    o: str

class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    year_limit: int = 5 # Default to 5 years

class AppraiseRequest(BaseModel):
    article_id: str
    title: str
    abstract: str

class GenerateReportRequest(BaseModel):
    pico_query: str
    articles: list[dict] # Expected to have title, abstract, year, etc.

class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    text: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ExtractPicoRequest(BaseModel):
    messages: list[ChatMessage]

class ModifyStrategyRequest(BaseModel):
    original_query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to EBM Appraisal Tool API"}

@app.post("/api/search-strategy")
async def generate_search_strategy(pico: PicoRequest):
    strategy = []
    if pico.p: strategy.append(f'("{pico.p}"[MeSH Terms] OR {pico.p}[Title/Abstract])')
    if pico.i: strategy.append(f'("{pico.i}"[MeSH Terms] OR {pico.i}[Title/Abstract])')
    if pico.c: strategy.append(f'("{pico.c}"[MeSH Terms] OR {pico.c}[Title/Abstract])')
    if pico.o: strategy.append(f'("{pico.o}"[MeSH Terms] OR {pico.o}[Title/Abstract])')
    
    query = " AND ".join(strategy) if strategy else ""
    return {"query": query}

@app.post("/api/chat")
async def chat_with_ebm(req: ChatRequest, db: Session = Depends(get_db)):
    # Using the directly provided API key from the user
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"response": "系統未設定 Gemini API 金鑰，無法進行討論。請在環境變數中設定 GEMINI_API_KEY。"}
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        system_instructions = """你是一位專業的實證醫學（EBM）助理。
你的任務是透過對話幫助使用者釐清他們的臨床問題，並整理出對應的 PICO 結構（Patient/Problem, Intervention, Comparison, Outcome）。
請用繁體中文回覆，語氣親切專業。
一開始先詢問使用者想要探討什麼樣的臨床情境或案例。
在對話過程中，可以適時提示使用者補充 PICO 的哪個要素還不夠清楚。
當你覺得 PICO 資訊已經足夠時，可以告訴使用者「看起來資訊很完整了，您可以點擊『抽出 PICO』來繼續下一個步驟」。"""

        # Prepare messages excluding the last one which will be sent via `send_message`
        messages_to_process = req.messages[:-1]
        
        # History must start with 'user' message
        while messages_to_process and messages_to_process[0].role != "user":
            messages_to_process = messages_to_process[1:]

        formatted_history = []
        for msg in messages_to_process:
            role = "user" if msg.role == "user" else "model"
            if not formatted_history:
                formatted_history.append({"role": role, "parts": [msg.text]})
            else:
                if formatted_history[-1]["role"] == role:
                    # Squash consecutive messages of the same role
                    formatted_history[-1]["parts"][0] += "\n\n" + msg.text
                else:
                    formatted_history.append({"role": role, "parts": [msg.text]})
        
        current_msg_text = req.messages[-1].text
        
        # History must end with 'model' so the new message sent is 'user'
        if formatted_history and formatted_history[-1]["role"] == "user":
            last_user_msg = formatted_history.pop()
            current_msg_text = last_user_msg["parts"][0] + "\n\n" + current_msg_text

        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=system_instructions
        )
        chat = model.start_chat(history=formatted_history)
        response = chat.send_message(current_msg_text)
        
        print(f"==========\\n[用戶紀錄] AI 對話:\\n用戶: {current_msg_text}\\n助理: {response.text}\\n==========")
        return {"response": response.text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Chat error: {e}")
        return {"response": f"抱歉，系統目前遇到了一點問題，請稍候再試。錯誤明細：{str(e)}"}

@app.post("/api/extract-pico")
async def extract_pico(req: ExtractPicoRequest, db: Session = Depends(get_db)):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"p": "", "i": "", "c": "", "o": ""}
        
    import google.generativeai as genai
    import json
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel(
            'gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )
        
        conversation_text = "\n".join([f"{msg.role}: {msg.text}" for msg in req.messages])
        
        prompt = f"""根據以下使用者與助理的對話紀錄，請幫我萃取出 PICO (Patient, Intervention, Comparison, Outcome) 四個要素，並將其轉換成英文的檢索關鍵字。
        
對話紀錄：
{conversation_text}

請回傳精確的 JSON 格式，包含 p, i, c, o 四個屬性。如果某個要素在對話中沒有提到，請留空字串。
所有的要素請盡量轉成英文，以便用於後續的 PubMed 搜尋。
例如: {{"p": "Type 2 Diabetes", "i": "Metformin", "c": "Placebo", "o": "Mortality"}}
"""
        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        pico_json = json.loads(text)
        print(f"==========\\n[用戶紀錄] 萃取 PICO:\\n{json.dumps(pico_json, ensure_ascii=False, indent=2)}\\n==========")
        return pico_json
    except Exception as e:
        print(f"PICO extraction error: {e}")
        return {"p": "", "i": "", "c": "", "o": ""}

@app.post("/api/auto-modify-strategy")
async def auto_modify_strategy(req: ModifyStrategyRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"new_query": req.original_query}
        
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""以下這段 PubMed 搜尋策略無法找到任何文獻，可能設限太嚴格或關鍵字不精確：

原搜尋策略：{req.original_query}

請幫我修改這段搜尋策略，使其放寬條件（例如拿掉比較組(Comparison)、拿掉預期結果(Outcome)、或是使用更廣泛的 MeSH terms 或同義字），來增加找到文獻的機會。
請「只」回傳修改後的搜尋策略字串，不要加上任何其他說明或引號。"""

        response = model.generate_content(prompt)
        new_query = response.text.strip()
        # Remove markdown code block if present
        if new_query.startswith("```"):
            new_query = "\n".join(new_query.split("\n")[1:-1]).strip()
        
        return {"new_query": new_query}
    except Exception as e:
        print(f"Auto modify strategy error: {e}")
        return {"new_query": req.original_query}

@app.post("/api/search")
async def search_pubmed(req: SearchRequest, db: Session = Depends(get_db)):
    print(f"==========\\n[用戶紀錄] 執行 PubMed 搜尋:\\n搜尋字串: {req.query}\\n限制年份: {req.year_limit} 年\\n==========")
    if not req.query:
        return {"results": []}
        
    base_esearch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": req.query,
        "retmode": "json",
        "retmax": req.max_results
    }
    
    if req.year_limit > 0:
        params["datetype"] = "pdat"
        params["reldate"] = req.year_limit * 365 # Approximate days
    
    async with httpx.AsyncClient() as client:
        search_res = await client.get(base_esearch, params=params)
        search_data = search_res.json()
        
        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return {"results": []}
            
        # Fetch detailed abstracts using efetch
        base_efetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        fetch_res = await client.get(base_efetch, params=fetch_params)
        
        results = []
        try:
            root = ET.fromstring(fetch_res.text)
            for article in root.findall('.//PubmedArticle'):
                pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else ""
                
                article_node = article.find('.//Article')
                title = article_node.find('.//ArticleTitle').text if article_node is not None and article_node.find('.//ArticleTitle') is not None else ""
                
                abstract_node = article_node.find('.//AbstractText') if article_node is not None else None
                abstract = abstract_node.text if abstract_node is not None and abstract_node.text else "No abstract available."
                
                pub_date = article_node.find('.//PubDate/Year') if article_node is not None else None
                year = pub_date.text if pub_date is not None else ""
                
                authors = []
                author_list = article.findall('.//Author')
                for author in author_list:
                    last_name = author.find('LastName')
                    fore_name = author.find('ForeName')
                    name = ""
                    if last_name is not None and last_name.text:
                        name = last_name.text
                        if fore_name is not None and fore_name.text:
                            name = name + f" {fore_name.text[0]}."
                    if name:
                        authors.append(name)
                
                results.append({
                    "id": pmid,
                    "title": title,
                    "authors": ", ".join(authors) if authors else "Unknown",
                    "year": year,
                    "abstract": abstract,
                    "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        except Exception as e:
            print("XML Parsing error:", e)
            
        return {"results": results}

@app.post("/api/appraise")
async def appraise_article(req: AppraiseRequest):
    # Retrieve GEMINI_API_KEY from environment variables
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        try:
             model = genai.GenerativeModel('gemini-2.5-flash')
             prompt = f"""You are a medical evidence appraiser using CASP guidelines.
             Please critically appraise the following abstract using formatting suitable for a web page.
             Title: {req.title}
             Abstract: {req.abstract}"""
             
             response = model.generate_content(prompt)
             llm_appraisal = response.text
             
             # Convert markdown-like response to simple HTML if needed,
             # or simply return the text wrapped in a pre/div tag.
             # The frontend uses dangerouslySetInnerHTML, so returning HTML is best.
             import markdown
             try:
                 html_content = markdown.markdown(llm_appraisal)
             except ImportError:
                 # fallback if markdown not installed
                 html_content = f"<pre style='white-space: pre-wrap; font-family: inherit;'>{llm_appraisal}</pre>"
                 
             return {"appraisal_html": f"<div class='gemini-response'>{html_content}</div>"}
        except Exception as e:
            print("Gemini API Error:", e)
            pass # fallback to mock
            
    # Mock appraisal
    html = f'''
        <h3>CASP 隨機對照試驗評讀模版 (自動分析結果)</h3>
        <div style="background: var(--bg-color); padding: 1.5rem; border-radius: 8px; margin-top: 1rem; border: 1px solid var(--border-color);">
            <h4>1. 試驗是否探討一個明確的問題？</h4>
            <p style="color: green; font-weight: 500;">✓ 可能是 (Yes/Unclear)</p>
            <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1rem;">基於摘要：{req.abstract[0:100] if req.abstract else ""}...</p>
            
            <h4>2. 是否提供適當的隨機分配？</h4>
            <p style="color: orange; font-weight: 500;">? 不明確 (Unclear)</p>
            <p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1rem;">摘要中未明確提供隨機分配細節。</p>
        </div>
        <p style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted);">*此為系統根據摘要自動生成之參考，請仔細閱讀全文以獲得準確評讀。</p>
    '''
    return {"appraisal_html": html}

@app.post("/api/generate-report")
async def generate_ebm_report(req: GenerateReportRequest, db: Session = Depends(get_db)):
    print(f"==========\\n[用戶紀錄] 生成綜合報告:\\n- PICO 搜尋字串: {req.pico_query}\\n- 納入文獻數量: {len(req.articles) if req.articles else 0} 篇\\n==========")
    if not req.articles:
        raise HTTPException(status_code=400, detail="未提供任何文獻進行分析。")

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        try:
             model = genai.GenerativeModel('gemini-2.5-flash')
             
             # Prepare the context from articles
             articles_context = ""
             for i, article in enumerate(req.articles):
                 articles_context += f"【文獻 {i+1}】\n標題: {article.get('title')}\n年份: {article.get('year')}\n摘要: {article.get('abstract')}\n\n"
             
             prompt = f"""請您扮演一位專業的實證醫學 (EBM) 專家。
             根據以下提供的 PICO 搜尋策略與納入分析的文獻，請以嚴謹的學術結構，使用繁體中文撰寫一份綜合實證分析報告。
             
             【提問與搜尋策略】
             {req.pico_query}
             
             【納入分析的文獻 (共 {len(req.articles)} 篇)】
             {articles_context}
             
             【報告必須包含以下五大區塊，並使用 Markdown 格式排版】
             1. 背景 (Background)：簡述本次臨床提問的核心內容。
             2. 方法 (Methods)：說明使用了哪些搜尋策略，以及最終納入了幾篇文獻進行分析。
             3. 評讀與結果 (Results)：必須使用 **CASP 評讀表格 (Markdown Table)** 來總結所有納入文獻的品質評估，並在表格下詳細補充說明每篇文獻的結果差異與實證等級。
             4. 結論 (Conclusion)：根據評讀結果，給出具體、可行的臨床建議。
             5. 參考文獻 (References)：列出所有納入分析的文獻，必須嚴格遵守 **APA 第七版 (APA 7th edition)** 格式。
             
             ⚠️ 重要要求：整份報告的內文撰寫（包含背景、方法、結果、結論），若有提及特定文獻，請務必使用標準的 **APA 文內引用 (In-text citations，例如：Author, Year)** 格式。
             """
             
             response = model.generate_content(prompt)
             llm_report = response.text
             
             import markdown
             try:
                 html_content = markdown.markdown(llm_report, extensions=['extra'])
             except ImportError:
                 html_content = f"<pre style='white-space: pre-wrap; font-family: inherit;'>{llm_report}</pre>"
                 
             return {"report_html": f"<div class='gemini-response ebm-report'>{html_content}</div>"}
        except Exception as e:
            print("Gemini API Error in generating report:", e)
            return {"report_html": "<p style='color: red;'>生成報告時發生錯誤，請檢查 API_KEY 或網路連線。</p>"}
            
    # Mock report if no API key
    mock_html = f'''
        <div class="ebm-report" style="background: var(--bg-color); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color);">
            <h2>實證醫學綜合分析報告 (測試摘要)</h2>
            <h3>1. 背景 (Background)</h3>
            <p>本報告旨在探討您的 PICO 提問。</p>
            <h3>2. 方法 (Methods)</h3>
            <p>本次分析共納入了 {len(req.articles)} 篇文獻。使用了相關的搜尋策略。</p>
            <h3>3. 結果 (Results)</h3>
            <ul>
                {"".join([f"<li>文獻 {i+1}: {a.get('title')} - ({a.get('year')})</li>" for i, a in enumerate(req.articles)])}
            </ul>
            <p>礙於系統未配置 Gemini API Key，無法進行深度語意分析。請在後端設定金鑰以啟用完整功能。</p>
            <h3>4. 結論 (Conclusion)</h3>
            <p>需根據完整文獻內容給出臨床建議。</p>
        </div>
    '''
    return {"report_html": mock_html}


@app.get("/admin_logs", response_class=HTMLResponse)
def view_admin_logs(db: Session = Depends(get_db)):
    try:
        logs = db.query(ActivityLog).order_by(ActivityLog.timestamp.desc()).limit(100).all()
        
        html = """
        <html>
            <head>
                <meta charset="utf-8">
                <title>EBM Tool 系統日誌後台</title>
                <style>
                    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background-color: #f7f9fc; }
                    h1 { color: #333; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                    th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #4CAF50; color: white; }
                    tr:hover { background-color: #f1f1f1; }
                    .details { white-space: pre-wrap; font-size: 0.9em; color: #555; }
                </style>
            </head>
            <body>
                <h1>EBM Tool 系統日誌後台 (最近 100 筆)</h1>
                <table>
                    <tr>
                        <th>時間 (UTC)</th>
                        <th>動作類型</th>
                        <th>詳細內容</th>
                    </tr>
        """
        for log in logs:
            time_str = log.timestamp.strftime("%Y-%m-%d %H:%M:%S") if log.timestamp else ""
            # Escape HTML to prevent XSS
            safe_details = log.details.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") if log.details else ""
            html += f"<tr><td>{time_str}</td><td>{log.action_type}</td><td class='details'>{safe_details}</td></tr>"
            
        html += """
                </table>
            </body>
        </html>
        """
        return html
    except Exception as e:
        return f"Database error: {e}"

if __name__ == "__main__":

    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# trigger reload
