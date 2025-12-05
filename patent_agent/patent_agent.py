import os
import json
import requests
import operator
import pandas as pd
import matplotlib.pyplot as plt
import base64
from fpdf import FPDF
from io import BytesIO
from typing import TypedDict, List, Annotated

# --- IMPORTS ---
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# --- FREE LOCAL EMBEDDINGS ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CHAT MODELS (Uncomment the one you can use) ---
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama  # Updated import for local LLM

# --- CONFIGURATION ---
# os.environ["OPENAI_API_KEY"] = "sk-..." 

# =============================================================================
# 1. RAG SETUP (Using Free Local Embeddings)
# =============================================================================
def initialize_vector_store():
    """Ingests patent_data.json using FREE local embeddings."""
    try:
        with open("patent_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: patent_data.json not found. Please create it first.")
        return None

    docs = []
    for p in data:
        content = f"Title: {p['title']}\nAbstract: {p['abstract']}\nClaims: {p['claims']}"
        meta = {"patent_id": p["patent_id"], "assignee": p["assignee"]}
        docs.append(Document(page_content=content, metadata=meta))

    print("--- ⚡ Generating Embeddings Locally (Free - HuggingFace) ---")
    # This runs on your CPU and requires NO API Key
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Initialize DB
vector_db = initialize_vector_store()

# =============================================================================
# 2. DEFINE TOOLS
# =============================================================================
@tool
def semantic_patent_search(query: str):
    """
    Useful for finding patents by description, mechanism of action, or drug name.
    """
    if not vector_db:
        return "Error: Database not initialized."
    
    results = vector_db.similarity_search(query, k=2)
    formatted_results = []
    for doc in results:
        formatted_results.append(f"ID: {doc.metadata['patent_id']}\nTitle: {doc.page_content}")
    return "\n---\n".join(formatted_results)

@tool
def check_patent_status(patent_id: str):
    """
    Checks legal status/expiry via Mock API.
    """
    try:
        # Connects to your running Flask API
        response = requests.get(f"http://127.0.0.1:5000/api/patents/status?patent_id={patent_id}")
        if response.status_code == 200:
            return str(response.json())
        else:
            return f"Error: Could not find status for {patent_id}"
    except Exception as e:
        return f"API Connection Error: {str(e)}"


@tool
def get_patent_landscape(molecule: str):
    """
    Calls mock API endpoint /api/patents/landscape and returns JSON.
    """
    try:
        response = requests.get(f"http://127.0.0.1:5000/api/patents/landscape?molecule={molecule}")
        if response.status_code == 200:
            return response.json()
        return f"API Error: {response.status_code}"
    except Exception as e:
        return f"API Error: {str(e)}"


@tool
def generate_status_table(molecule: str):
    """
    Generates a CSV status table for patents for a given molecule and saves to ./output.
    Returns the CSV path.
    """
    try:
        js = get_patent_landscape.invoke({'molecule': molecule})
        if isinstance(js, str) and js.startswith('API Error'):
            return js
        patents = js.get('patents', [])
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        df = pd.DataFrame([{
            'patent_id': p['patent_id'],
            'title': p['title'],
            'assignee': p['assignee'],
            'status': p['status'],
            'expiry_date': p['expiry_date'],
            'filing_date': p.get('filing_date', ''),
            'indication': p.get('indication', ''),
            'cpc_codes': ','.join(p.get('cpc_codes', []))
        } for p in patents])
        # compute expiry / FTO columns
        df['is_expired'] = df['expiry_date'].apply(lambda d: True if d=='N/A' or d < today else False)
        df['is_generic_opportunity'] = df['is_expired']
        os.makedirs('output', exist_ok=True)
        csv_path = os.path.join('output', f'patent_status_{molecule}.csv')
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        return f"Error generating CSV: {str(e)}"


@tool
def generate_filings_heatmap(molecule: str):
    """
    Generates a filings chart (PNG) showing filings per year for the molecule.
    Returns base64 string and the image path.
    """
    try:
        response = requests.get(f"http://127.0.0.1:5000/api/patents/filings?molecule={molecule}")
        if response.status_code != 200:
            return f"API Error: {response.status_code}"
        js = response.json()
        filings = js.get('filings_by_year', {})
        if not filings:
            return "No filings data available"

        years = sorted(filings.keys())
        counts = [filings[y] for y in years]

        plt.figure(figsize=(8,4))
        plt.bar(years, counts, color='tab:blue')
        plt.xlabel('Year')
        plt.ylabel('Number of Filings')
        plt.title(f'Patent Filings by Year - {molecule.title()}')
        plt.tight_layout()
        os.makedirs('output', exist_ok=True)
        img_path = os.path.join('output', f'filings_{molecule}.png')
        plt.savefig(img_path)
        plt.close()

        # Return base64 representation
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')

        return {'image_path': img_path, 'base64': b64}
    except Exception as e:
        return f"Error generating filings heatmap: {str(e)}"


@tool
def export_patent_pdf(patent_id: str):
    """
    Export the patent details to a PDF file in ./output
    Returns the path to the PDF.
    """
    try:
        # Use search to find the patent
        response = requests.get(f"http://127.0.0.1:5000/api/patents/search?q={patent_id}")
        if response.status_code != 200:
            return f"API Error: {response.status_code}"
        data = response.json()
        patents = data.get('patents', [])
        if not patents:
            return f"Patent not found: {patent_id}"
        p = next((pat for pat in patents if pat['patent_id'] == patent_id), patents[0])

        os.makedirs('output', exist_ok=True)
        pdf_path = os.path.join('output', f'{patent_id}.pdf')
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 8, f"Patent {p['patent_id']}", ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 6, f"Title: {p['title']}\nAssignee: {p.get('assignee', '')}\nStatus: {p.get('status', '')}\nFiling Date: {p.get('filing_date', '')}\nExpiry Date: {p.get('expiry_date', '')}\nCPC Codes: {', '.join(p.get('cpc_codes', []))}\n\nAbstract:\n{p.get('abstract', '')}\n\nClaims:\n{p.get('claims', '')}")
        pdf.output(pdf_path)
        return pdf_path
    except Exception as e:
        return f"Error exporting PDF: {str(e)}"


@tool
def generate_patent_report(molecule: str, top_n: int = 3):
    """
    Generate a consolidated patent report: CSV status table, filings heatmap, and PDF extracts for top_n patents.
    Returns a dict with paths to the generated artifacts.
    """
    try:
        report = {}
        csv_path = generate_status_table.invoke(molecule)
        report['csv'] = csv_path

        heatmap_info = generate_filings_heatmap.invoke(molecule)
        if isinstance(heatmap_info, dict):
            report['heatmap_path'] = heatmap_info.get('image_path')
        else:
            report['heatmap_path'] = None

        js = get_patent_landscape.invoke({'molecule': molecule})
        patents = js.get('patents', []) if isinstance(js, dict) else []
        # Choose top_n patents to export - prioritize Active then Pending
        patents_sorted = sorted(patents, key=lambda p: (p.get('status') != 'Active', p.get('expiry_date', '')))
        pdf_paths = []
        for p in patents_sorted[:top_n]:
            pid = p['patent_id']
            pdfp = export_patent_pdf.invoke(pid)
            pdf_paths.append(pdfp)
        report['pdfs'] = pdf_paths
        return report
    except Exception as e:
        return f"Error generating patent report: {str(e)}"

tools = [semantic_patent_search, check_patent_status, get_patent_landscape, generate_status_table, generate_filings_heatmap, export_patent_pdf, generate_patent_report]

# =============================================================================
# 3. DEFINE AGENT STATE
# =============================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# =============================================================================
# 4. DEFINE NODES
# =============================================================================
def agent_node(state: AgentState):
    """The reasoning engine."""
    messages = state['messages']
    
    # --- CHOOSE YOUR LLM HERE ---
    
    # OPTION A: OpenAI (Try this first)
    # llm = ChatOpenAI(model="gpt-4", temperature=0)

    # OPTION B: Ollama (FREE/LOCAL) - Use this if OpenAI gives "Insufficient Quota" again
    # You must install Ollama and run 'ollama run mistral' first
    llm = ChatOllama(model="mistral", temperature=0)

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    try:
        response = llm_with_tools.invoke(messages)
    except Exception as e:
        return {"messages": [HumanMessage(content=f"LLM Error: {str(e)}. Check your API Key or Quota.")]}

    return {"messages": [response]}

def tool_node(state: AgentState):
    """Executes the requested tools."""
    last_message = state['messages'][-1]
    
    if not last_message.tool_calls:
        return {"messages": []}
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        if tool_name == "semantic_patent_search":
            arg = tool_args.get('query') if isinstance(tool_args, dict) and 'query' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = semantic_patent_search.invoke(arg)
        elif tool_name == "check_patent_status":
            arg = tool_args.get('patent_id') if isinstance(tool_args, dict) and 'patent_id' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = check_patent_status.invoke(arg)
        elif tool_name == "get_patent_landscape":
            arg = tool_args.get('molecule') if isinstance(tool_args, dict) and 'molecule' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = get_patent_landscape.invoke(arg)
        elif tool_name == "generate_status_table":
            arg = tool_args.get('molecule') if isinstance(tool_args, dict) and 'molecule' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = generate_status_table.invoke(arg)
        elif tool_name == "generate_filings_heatmap":
            arg = tool_args.get('molecule') if isinstance(tool_args, dict) and 'molecule' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = generate_filings_heatmap.invoke(arg)
        elif tool_name == "export_patent_pdf":
            arg = tool_args.get('patent_id') if isinstance(tool_args, dict) and 'patent_id' in tool_args else next(iter(tool_args.values())) if isinstance(tool_args, dict) and tool_args else tool_args
            res = export_patent_pdf.invoke(arg)
        elif tool_name == "generate_patent_report":
            # Accept either a dict {'molecule': 'metformin', 'top_n': 3} or a single molecule string
            if isinstance(tool_args, dict):
                mol = tool_args.get('molecule') or next(iter(tool_args.values()))
                top_n = int(tool_args.get('top_n', 3))
            else:
                mol, top_n = tool_args, 3
            res = generate_patent_report.invoke({'molecule': mol, 'top_n': top_n})
        else:
            res = "Error: Unknown tool."
            
        results.append(ToolMessage(
            tool_call_id=tool_call['id'], 
            name=tool_name, 
            content=str(res)
        ))
        
    return {"messages": results}

def router(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# =============================================================================
# 5. BUILD GRAPH
# =============================================================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# =============================================================================
# 6. EXECUTION
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Patent Agent - CLI')
    parser.add_argument('--action', '-a', choices=['query', 'report'], default='query')
    parser.add_argument('--molecule', '-m', default='Metformin')
    parser.add_argument('--top_n', '-n', type=int, default=3)
    parser.add_argument('--query', '-q', default='Find active patents related to treating Alzheimer\'s with Metformin.')
    args = parser.parse_args()

    print("--- Patent Agent (Local Embeddings) Started ---")

    if args.action == 'report':
        print(f"Generating report for molecule: {args.molecule} (top_n={args.top_n})...")
        report = generate_patent_report.invoke({'molecule': args.molecule.lower(), 'top_n': args.top_n})
        print('Report Artifacts:')
        print(report)
    else:
        query = args.query
        inputs = {"messages": [HumanMessage(content=query)]}
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Node '{key}':\n{value['messages'][-1]}\n")