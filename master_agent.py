import json
import operator
from typing import Annotated, List, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# Import worker functions
from iqvia_agent import iqvia_search
from exim_agent import exim_search
from patent_agent import patent_search
from clinical_agent import clinical_search
from web_agent import web_search

# --- Configuration ---
LLM_MODEL = "phi3"

# --- State Definition ---
class AgentState(TypedDict):
    input_query: str
    plan: list  # List of dicts: [{"agent": "iqvia", "query": "..."}]
    results: Annotated[List, operator.add]  # Accumulates results from workers (now dicts)
    final_answer: str
    visuals: list  # List of visual objects for charts/tables
    skip_pipeline: bool  # Flag to skip full pipeline for simple queries
    preflight_response: str  # Response from preflight node for greetings/identity
    memory_context: str  # Conversation memory context for continuity
    is_follow_up: bool  # Whether this is a follow-up query
    key_topics: list  # Key topics from conversation history

# --- LLM Setup ---
llm = ChatOllama(model=LLM_MODEL, format="json")
llm_text = ChatOllama(model=LLM_MODEL) # Non-json mode for final answer

# --- Node: Preflight (handles greetings, identity, simple queries) ---
def preflight_node(state: AgentState):
    """Detects if query is a greeting, identity question, or simple query that doesn't need full pipeline."""
    import re
    
    query = state['input_query'].lower().strip()
    # Normalize common variations
    query_normalized = query.replace("'", "").replace("?", "").replace("!", "").strip()
    query_words = set(query_normalized.split())  # Split into words for exact matching
    
    print(f"\n[Preflight] Checking query: '{query[:50]}...' " if len(query) > 50 else f"\n[Preflight] Checking query: '{query}'")
    
    # FIRST: Check if query has pharma/data context - if so, skip greetings check
    pharma_keywords = [
        'drug', 'medicine', 'pharma', 'clinical', 'trial', 'trials', 'patent', 'patents',
        'market', 'import', 'export', 'api', 'sales', 'iqvia', 'exim', 'cancer', 'diabetes',
        'cardio', 'neuro', 'immuno', 'oncology', 'vaccine', 'therapeutic', 'fda',
        'approval', 'pipeline', 'competitor', 'generic', 'biosimilar', 'molecule', 'data',
        'show', 'tell', 'give', 'find', 'search', 'list', 'get', 'fetch', 'display',
        'news', 'latest', 'recent', 'article', 'web', 'look up', 'lookup', 'information'
    ]
    
    has_pharma_context = any(kw in query_normalized for kw in pharma_keywords)
    
    # If the query has pharma context, proceed with full pipeline regardless of greetings
    if has_pharma_context:
        print("[Preflight] Pharma context detected, proceeding with full pipeline")
        return {"skip_pipeline": False, "preflight_response": ""}
    
    # Greeting patterns (expanded) - these need exact word match
    greetings = [
        'hello', 'hi', 'hey', 'howdy', 'greetings', 'yo', 'sup', 'hiya', 'heya'
    ]
    
    # Multi-word greetings
    greeting_phrases = [
        'good morning', 'good afternoon', 'good evening', 'good day'
    ]
    
    # Casual conversation patterns
    casual_patterns = [
        'whats up', 'what is up', 'what up', 'wassup', 'wazzup',
        'how are you', 'how r u', 'hows it going', 'how is it going',
        'whats going on', 'what is going on', 'whats new', 'what is new',
        'how do you do', 'hows your day', 'how is your day',
        'nice to meet you', 'pleased to meet you'
    ]
    
    identity_questions = [
        'who are you', 'what are you', 'what is your name', 'whats your name',
        'what can you do', 'what do you do', 'what is gloser', 'whats gloser',
        'tell me about yourself', 'introduce yourself'
    ]
    
    thanks = ['thank you', 'thanks', 'thx', 'appreciated', 'cheers', 'ty']
    
    farewells = ['bye', 'goodbye', 'see you', 'later', 'take care', 'cya', 'ttyl']
    
    # Check single-word greetings (must be exact word match, not substring)
    for greeting in greetings:
        if greeting in query_words or query_normalized == greeting:
            print("[Preflight] Detected greeting, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "Hello! I'm Gloser AI, your pharmaceutical intelligence assistant. I can help you analyze market data, clinical trials, patents, and import/export trade information. What would you like to know?"
            }
    
    # Check multi-word greeting phrases
    for phrase in greeting_phrases:
        if query_normalized.startswith(phrase):
            print("[Preflight] Detected greeting phrase, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "Hello! I'm Gloser AI, your pharmaceutical intelligence assistant. I can help you analyze market data, clinical trials, patents, and import/export trade information. What would you like to know?"
            }
    
    # Check casual patterns
    for casual in casual_patterns:
        if casual in query_normalized or query_normalized == casual:
            print("[Preflight] Detected casual conversation, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "Hey there! I'm doing great, thanks for asking! 😊\n\nI'm Gloser AI, ready to help you with pharmaceutical market intelligence. I can analyze:\n\n• **Clinical Trials** - Find trials by drug, phase, or indication\n• **Market Data** - Market sizes, growth rates, competition\n• **Patents** - Patent status, expiry dates, assignees\n• **Trade Data** - Import/export volumes for APIs\n\nWhat would you like to explore today?"
            }
    
    # Check identity questions
    for identity in identity_questions:
        if identity in query_normalized:
            print("[Preflight] Detected identity question, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "I'm Gloser AI, a pharmaceutical market intelligence platform. I can help you with:\n\n• **Market Analysis** - Market sizes, CAGR, competitors by therapeutic area\n• **Clinical Trials** - Trial phases, sponsors, recruitment status\n• **Patent Landscape** - Patent filings, expiry dates, assignees\n• **Trade Data** - Import/export volumes for pharmaceutical APIs\n\nJust ask me anything about the pharmaceutical industry!"
            }
    
    # Check thanks (exact word match)
    for thank in thanks:
        if thank in query_words or thank in query_normalized:
            print("[Preflight] Detected thank you, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "You're welcome! Let me know if you have any other questions about the pharmaceutical market."
            }
    
    # Check farewells (exact word match)
    for farewell in farewells:
        if farewell in query_words or query_normalized == farewell:
            print("[Preflight] Detected farewell, skipping pipeline")
            return {
                "skip_pipeline": True,
                "preflight_response": "Goodbye! Feel free to come back anytime you need pharmaceutical market insights. Take care! 👋"
            }
    
    # Check if query is too short/vague to be a data query (less than 3 meaningful words)
    words = [w for w in query_normalized.split() if len(w) > 2]
    
    if len(words) < 2:
        print("[Preflight] Query too vague, asking for clarification")
        return {
            "skip_pipeline": True,
            "preflight_response": f"I'd be happy to help! Could you please be more specific about what pharmaceutical information you're looking for?\n\nFor example, you could ask:\n• \"Show me clinical trials for diabetes\"\n• \"What's the market size for oncology drugs?\"\n• \"Find patents for metformin\"\n• \"Export data for paracetamol API\""
        }
    
    # Not a simple query - proceed with full pipeline
    print("[Preflight] Not a simple query, proceeding with full pipeline")
    return {"skip_pipeline": False, "preflight_response": ""}

# --- Node: Planner ---
def planner_node(state: AgentState):
    #print(f"\n[Master] Planning for query: {state['input_query']}")
    
    # Build memory context section for prompt
    memory_section = ""
    if state.get('memory_context'):
        memory_section = f"""
    CONVERSATION CONTEXT (use this to understand follow-up queries):
    {state['memory_context']}
    ---
    """
    
    # Add follow-up hint if detected
    follow_up_hint = ""
    if state.get('is_follow_up'):
        follow_up_hint = """
    NOTE: This appears to be a follow-up question. Consider the previous conversation context when planning.
    If the user refers to "it", "that", "same", etc., infer from the conversation history what they mean.
    """
    
    # Add key topics hint
    topics_hint = ""
    if state.get('key_topics'):
        topics_hint = f"""
    Key topics from conversation: {', '.join(state['key_topics'])}
    """
    
    prompt = f"""
    You are a Master Orchestrator for pharmaceutical intelligence.
    {memory_section}
    {follow_up_hint}
    {topics_hint}
    User Query: "{state['input_query']}"
    
    Available Agents:
    1. iqvia - Market share, sales data, therapeutic area analysis (use for: market size, competition, CAGR, industry trends)
    2. exim - Export/import trade data, shipping volumes (use for: trade data, API imports/exports, country-wise trade)
    3. patent - Patent filings, expiry dates, assignees (use for: patent status, IP information, molecule patents)
    4. clinical - Clinical trials from ClinicalTrials.gov (use for: trial phases, sponsors, recruitment status)
    5. web - Web search for LATEST news, articles, recent developments (use for: news, recent events, company announcements, FDA updates, anything not in our databases)
    
    IMPORTANT GUIDELINES:
    - For vague/broad queries (like just "patent" or "market"), use multiple relevant agents including web for latest context
    - ALWAYS include web agent when user asks about "latest", "recent", "news", "updates", or "more data"
    - ALWAYS include web agent when user wants comprehensive/additional information beyond our databases
    - When user says "more data" or "anything else", add web agent to search for supplementary information
    - Web agent is great for real-time news, company announcements, FDA approvals, and industry developments
    
    Return a JSON object with a key "steps" containing a list of agents to call and the specific query for them.
    Format:
    {{
      "steps": [
        {{"agent": "iqvia", "query": "specific keyword"}},
        {{"agent": "exim", "query": "specific keyword"}},
        {{"agent": "patent", "query": "specific keyword"}},
        {{"agent": "clinical", "query": "specific keyword"}},
        {{"agent": "web", "query": "specific search query for latest news/info"}}
      ]
    }}
    Return only agents that are relevant to the user query. When in doubt, include web agent for additional context.
    If this is a follow-up query, use context from previous conversation to determine the right agents and keywords.
    """
    
    response = llm.invoke(prompt)
    try:
        plan_json = json.loads(response.content)
        plan = plan_json.get("steps", [])
        print(f"\n[Planner] Generated plan with {len(plan)} step(s):")
        for step in plan:
            print(f"  - Agent: {step.get('agent')}, Query: {step.get('query')}")
        return {"plan": plan}
    except Exception as e:
        print(f"[Planner] Error parsing plan: {e}")
        print(f"[Planner] Raw LLM response: {response.content[:200]}")
        return {"plan": []}

# --- Node: Worker Wrappers ---
def iqvia_node(state: AgentState):
    # Find the query destined for this agent in the plan
    query = next((step['query'] for step in state['plan'] if step['agent'] == 'iqvia'), state['input_query'])
    print(f"\n[IQVIA Agent] Searching for: {query}")
    result = iqvia_search(query)
    print(f"[IQVIA Agent] Found {len(result.get('data', []))} records")
    print(f"[IQVIA Agent] Summary: {result.get('summary', 'N/A')}")
    return {"results": [result]}

def exim_node(state: AgentState):
    query = next((step['query'] for step in state['plan'] if step['agent'] == 'exim'), state['input_query'])
    print(f"\n[EXIM Agent] Searching for: {query}")
    result = exim_search(query)
    print(f"[EXIM Agent] Found {len(result.get('data', []))} records")
    print(f"[EXIM Agent] Summary: {result.get('summary', 'N/A')}")
    if result.get('data'):
        print(f"[EXIM Agent] Sample data keys: {list(result['data'][0].keys()) if result['data'] else 'None'}")
    return {"results": [result]}

def patent_node(state: AgentState):
    query = next((step['query'] for step in state['plan'] if step['agent'] == 'patent'), state['input_query'])
    print(f"\n[PATENT Agent] Searching for: {query}")
    result = patent_search(query)
    print(f"[PATENT Agent] Found {len(result.get('data', []))} records")
    print(f"[PATENT Agent] Summary: {result.get('summary', 'N/A')}")
    return {"results": [result]}

def clinical_node(state: AgentState):
    query = next((step['query'] for step in state['plan'] if step['agent'] == 'clinical'), state['input_query'])
    print(f"\n[CLINICAL Agent] Searching for: {query}")
    result = clinical_search(query)
    print(f"[CLINICAL Agent] Found {len(result.get('data', []))} records")
    print(f"[CLINICAL Agent] Summary: {result.get('summary', 'N/A')}")
    return {"results": [result]}

def web_node(state: AgentState):
    query = next((step['query'] for step in state['plan'] if step['agent'] == 'web'), state['input_query'])
    print(f"\n[WEB Agent] Searching for: {query}")
    result = web_search(query)
    print(f"[WEB Agent] Found {len(result.get('data', []))} results")
    print(f"[WEB Agent] Summary: {result.get('summary', 'N/A')}")
    return {"results": [result]}

# --- Visual Generation Helper ---
def generate_visuals(results: list, query: str) -> list:
    """Generate chart/table visuals from agent results."""
    print(f"\n[Visual Generator] Processing {len(results)} result(s)")
    visuals = []
    
    for result in results:
        if isinstance(result, str):
            print(f"[Visual Generator] Skipping string result: {result[:50]}...")
            continue  # Skip string results
            
        agent = result.get("agent", "")
        data = result.get("data", [])
        print(f"[Visual Generator] Agent: {agent}, Data records: {len(data)}")
        
        if not data:
            print(f"[Visual Generator] No data for {agent}, skipping...")
            continue
            
        if agent == "exim":
            print(f"[Visual Generator] Processing EXIM data...")
            # Generate visuals for EXIM data
            for item in data:
                drug_name = item.get("drug_name", item.get("hs_desc", item.get("drug", "Unknown")))
                print(f"[Visual Generator] EXIM item: {drug_name}")
                
                # Handle nested country_data structure
                country_info = None
                if "country_data" in item:
                    print(f"[Visual Generator] Found nested country_data structure")
                    # Get first country's data (typically India)
                    for country, info in item["country_data"].items():
                        country_info = info
                        print(f"[Visual Generator] Using country: {country}")
                        break
                
                # Use country_info if available, otherwise use item directly
                source_data = country_info if country_info else item
                
                # Pie chart for top import sources
                sources = source_data.get("top_import_sources", [])
                print(f"[Visual Generator] EXIM sources found: {len(sources)}")
                if sources:
                    print(f"[Visual Generator] Creating pie chart for import sources")
                    visuals.append({
                        "type": "pie",
                        "title": f"Top Import Sources for {drug_name}",
                        "labels": [s.get("country", "Unknown") for s in sources],
                        "datasets": [{
                            "data": [s.get("percentage", s.get("percent", 0)) for s in sources],
                            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                        }]
                    })
                
                # Line chart for yearly trend
                yearly_trend = source_data.get("yearly_trend", {})
                print(f"[Visual Generator] EXIM yearly_trend type: {type(yearly_trend).__name__}, data: {yearly_trend}")
                if yearly_trend:
                    # Handle dict format: {"2019": 4100, "2020": 4300, ...}
                    if isinstance(yearly_trend, dict):
                        print(f"[Visual Generator] Creating line chart for yearly trend (dict format)")
                        years = sorted(yearly_trend.keys())
                        values = [yearly_trend[y] for y in years]
                        visuals.append({
                            "type": "line",
                            "title": f"Yearly Volume Trend for {drug_name}",
                            "labels": years,
                            "datasets": [{
                                "label": "Volume (MT)",
                                "data": values,
                                "borderColor": "#36A2EB",
                                "fill": False
                            }]
                        })
                    # Handle list format: [{"year": 2019, "import_mt": 100, "export_mt": 50}, ...]
                    elif isinstance(yearly_trend, list):
                        visuals.append({
                            "type": "line",
                            "title": f"Yearly Import/Export Trend for {drug_name}",
                            "labels": [str(t.get("year", "")) for t in yearly_trend],
                            "datasets": [
                                {
                                    "label": "Import (MT)",
                                    "data": [t.get("import_mt", 0) for t in yearly_trend],
                                    "borderColor": "#36A2EB",
                                    "fill": False
                                },
                                {
                                    "label": "Export (MT)",
                                    "data": [t.get("export_mt", 0) for t in yearly_trend],
                                    "borderColor": "#FF6384",
                                    "fill": False
                                }
                            ]
                        })
                
                # Table for trade data summary
                import_vol = source_data.get("import_volume_mt", item.get("import_volume_mt", "N/A"))
                export_vol = source_data.get("export_volume_mt", item.get("export_volume_mt", "N/A"))
                import_val = source_data.get("import_value_million_usd", "N/A")
                export_val = source_data.get("export_value_million_usd", "N/A")
                
                visuals.append({
                    "type": "table",
                    "title": f"Trade Summary for {drug_name}",
                    "columns": ["Metric", "Value"],
                    "rows": [
                        ["Drug Name", drug_name],
                        ["Category", item.get("category", "N/A")],
                        ["Import Volume (MT)", str(import_vol)],
                        ["Export Volume (MT)", str(export_vol)],
                        ["Import Value (USD M)", str(import_val)],
                        ["Export Value (USD M)", str(export_val)]
                    ]
                })
        
        elif agent == "iqvia":
            # Parse market size from string like "185 Billion" to number
            def parse_market_size(val):
                if isinstance(val, (int, float)):
                    return val
                if isinstance(val, str):
                    val = val.lower().replace(',', '').strip()
                    multiplier = 1
                    if 'billion' in val:
                        multiplier = 1000  # Convert to millions for display
                        val = val.replace('billion', '').strip()
                    elif 'million' in val:
                        val = val.replace('million', '').strip()
                    try:
                        return float(val.replace('$', '').strip()) * multiplier
                    except:
                        return 0
                return 0
            
            # Bar chart for market data comparison
            if len(data) > 0:
                areas = [d.get("area", d.get("therapeutic_area", "Unknown")) for d in data[:5]]
                market_sizes = [parse_market_size(d.get("market_size_usd", 0)) for d in data[:5]]
                
                if any(market_sizes):
                    visuals.append({
                        "type": "bar",
                        "title": "Market Size Comparison (USD Million)",
                        "labels": areas,
                        "datasets": [{
                            "label": "Market Size (M)",
                            "data": market_sizes,
                            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                        }]
                    })
                
                # Bar chart for growth rates
                growth_rates = [d.get("cagr_percent", d.get("growth_rate", 0)) for d in data[:5]]
                if any(growth_rates):
                    visuals.append({
                        "type": "bar",
                        "title": "Growth Rate Comparison (CAGR %)",
                        "labels": areas,
                        "datasets": [{
                            "label": "CAGR %",
                            "data": growth_rates,
                            "backgroundColor": "#4BC0C0"
                        }]
                    })
                
                # Table for IQVIA data
                visuals.append({
                    "type": "table",
                    "title": "Market Intelligence Data",
                    "columns": ["Therapeutic Area", "Market Size", "CAGR %", "Competition", "Key Trend"],
                    "rows": [
                        [
                            d.get("area", d.get("therapeutic_area", "N/A")),
                            d.get("market_size_usd", "N/A"),
                            f"{d.get('cagr_percent', d.get('growth_rate', 'N/A'))}%",
                            d.get("competition_level", "N/A"),
                            d.get("key_trend", "N/A")[:50] + "..." if len(d.get("key_trend", "")) > 50 else d.get("key_trend", "N/A")
                        ] for d in data[:10]
                    ]
                })
        
        elif agent == "patent":
            # Table for patent data
            if data:
                visuals.append({
                    "type": "table",
                    "title": "Patent Information",
                    "columns": ["Molecule", "Patent ID", "Status", "Expiry Date", "Assignee"],
                    "rows": [
                        [
                            d.get("molecule", d.get("drug_name", "N/A")),
                            d.get("patent_id", d.get("patent_number", "N/A")),
                            d.get("status", "N/A"),
                            d.get("expiry_date", "N/A"),
                            d.get("assignee", d.get("patent_owner", "N/A"))[:25] + "..." if len(d.get("assignee", d.get("patent_owner", ""))) > 25 else d.get("assignee", d.get("patent_owner", "N/A"))
                        ] for d in data[:10]
                    ]
                })
                
                # Bar chart for patent expiry timeline
                expiry_data = [(d.get("molecule", d.get("title", ""))[:15], d.get("expiry_date", "")) 
                               for d in data if d.get("expiry_date")]
                if expiry_data:
                    visuals.append({
                        "type": "bar",
                        "title": "Patent Expiry Timeline",
                        "labels": [d[0] for d in expiry_data[:5]],
                        "datasets": [{
                            "label": "Expiry Year",
                            "data": [int(d[1].split("-")[0]) if "-" in d[1] else 2025 for d in expiry_data[:5]],
                            "backgroundColor": "#9966FF"
                        }]
                    })
                
                # Pie chart for patent status distribution
                statuses = {}
                for d in data:
                    status = d.get("status", "Unknown")
                    statuses[status] = statuses.get(status, 0) + 1
                if statuses:
                    visuals.append({
                        "type": "pie",
                        "title": "Patent Status Distribution",
                        "labels": list(statuses.keys()),
                        "datasets": [{
                            "data": list(statuses.values()),
                            "backgroundColor": ["#10B981", "#EF4444", "#F59E0B", "#6366F1", "#8B5CF6"]
                        }]
                    })
        
        elif agent == "clinical":
            # Table for clinical trials
            if data:
                visuals.append({
                    "type": "table",
                    "title": "Clinical Trials Data",
                    "columns": ["Trial ID", "Title", "Phase", "Status", "Sponsor", "Country"],
                    "rows": [
                        [
                            d.get("NCTId", d.get("trial_id", d.get("nct_id", "N/A"))),
                            (d.get("BriefTitle", d.get("drug", d.get("drug_name", "N/A")))[:40] + "...") if len(d.get("BriefTitle", d.get("drug", d.get("drug_name", ""))) or "") > 40 else d.get("BriefTitle", d.get("drug", d.get("drug_name", "N/A"))),
                            d.get("Phase", d.get("phase", "N/A")) or "N/A",
                            d.get("OverallStatus", d.get("status", "N/A")),
                            (d.get("LeadSponsorName", d.get("sponsor", "N/A"))[:25] + "...") if len(d.get("LeadSponsorName", d.get("sponsor", "")) or "") > 25 else d.get("LeadSponsorName", d.get("sponsor", "N/A")),
                            d.get("LocationCountry", d.get("country", "N/A"))
                        ] for d in data[:10]
                    ]
                })
                
                # Pie chart for trial phases distribution
                phases = {}
                for d in data:
                    phase = d.get("Phase", d.get("phase", "Unknown")) or "Unknown"
                    phases[phase] = phases.get(phase, 0) + 1
                
                if phases:
                    visuals.append({
                        "type": "pie",
                        "title": "Clinical Trials by Phase",
                        "labels": list(phases.keys()),
                        "datasets": [{
                            "data": list(phases.values()),
                            "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"]
                        }]
                    })
                
                # Pie chart for trial status distribution
                statuses = {}
                for d in data:
                    status = d.get("OverallStatus", d.get("status", "Unknown")) or "Unknown"
                    statuses[status] = statuses.get(status, 0) + 1
                
                if statuses:
                    visuals.append({
                        "type": "pie",
                        "title": "Clinical Trials by Status",
                        "labels": list(statuses.keys()),
                        "datasets": [{
                            "data": list(statuses.values()),
                            "backgroundColor": ["#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6"]
                        }]
                    })
        
        elif agent == "web":
            # Web results are included in the text response only, no visual table needed
            pass
    
    print(f"[Visual Generator] Total visuals generated: {len(visuals)}")
    for i, v in enumerate(visuals):
        print(f"[Visual Generator] Visual {i+1}: type={v.get('type')}, title={v.get('title', 'N/A')[:40]}")
    return visuals


# --- Node: Synthesizer ---
def synthesizer_node(state: AgentState):
    # If preflight handled the query, use that response
    if state.get('skip_pipeline') and state.get('preflight_response'):
        print("[Synthesizer] Using preflight response (skipping pipeline)")
        return {"final_answer": state['preflight_response'], "visuals": []}
    
    print("\n" + "="*50)
    print("[Synthesizer] Starting synthesis...")
    results = state['results']
    print(f"[Synthesizer] Received {len(results)} result(s) from agents")
    
    # Build context from results (handle both dict and string formats)
    context_parts = []
    for r in results:
        if isinstance(r, dict):
            agent_name = r.get('agent', 'unknown').upper()
            summary = r.get('summary', json.dumps(r.get('data', [])))
            context_parts.append(f"{agent_name} Data:\n{summary}")
        else:
            context_parts.append(str(r))
    
    context = "\n\n".join(context_parts)
    
    # Build memory context section for the prompt
    memory_section = ""
    if state.get('memory_context'):
        memory_section = f"""
    CONVERSATION HISTORY (for context continuity):
    {state['memory_context']}
    ---
    """
    
    # Add follow-up context
    follow_up_instruction = ""
    if state.get('is_follow_up'):
        follow_up_instruction = """
    NOTE: This is a follow-up question. Reference the previous conversation naturally.
    Use phrases like "Continuing from our previous discussion..." or "As mentioned earlier..." when appropriate.
    """
    
    prompt = f"""
    {memory_section}
    {follow_up_instruction}
    User Query: {state['input_query']}
    Answer the user query based ONLY on the provided context.
    Use provided context from various agents.
    You must synthesize the information into a concise answer. 
    Also give more relevant insights if possible.
    If this is a follow-up question, ensure your response builds upon the previous conversation naturally.
    Context from Agents: 
    {context} 
    
    Provide a detailed answer and properly mention stats every time.
    """
    
    print("[Synthesizer] Invoking LLM for final answer...")
    response = llm_text.invoke(prompt)
    print(f"[Synthesizer] LLM response length: {len(response.content)} chars")
    
    # Generate visuals from agent results
    print("[Synthesizer] Generating visuals...")
    visuals = generate_visuals(results, state['input_query'])
    
    print(f"\n[Synthesizer] FINAL OUTPUT:")
    print(f"  - Answer length: {len(response.content)} chars")
    print(f"  - Visuals count: {len(visuals)}")
    print("="*50 + "\n")
    
    return {"final_answer": response.content, "visuals": visuals}

# --- Routing Logic ---
def route_preflight(state: AgentState):
    """Route from preflight - skip to synthesizer if simple query, else go to planner."""
    if state.get('skip_pipeline'):
        return "synthesizer"
    return "planner"

def route_step(state: AgentState):
    plan = state['plan']
    routes = []
    for step in plan:
        if step['agent'] == 'iqvia':
            routes.append("iqvia")
        elif step['agent'] == 'exim':
            routes.append("exim")
        elif step['agent'] == 'patent':
            routes.append("patent")
        elif step['agent'] == 'clinical':
            routes.append("clinical")
        elif step['agent'] == 'web':
            routes.append("web")
    
    if not routes:
        return "synthesizer" # If no agents needed, skip to end
    return routes

# --- Graph Construction ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("preflight", preflight_node)
workflow.add_node("planner", planner_node)
workflow.add_node("iqvia", iqvia_node)
workflow.add_node("exim", exim_node)
workflow.add_node("patent", patent_node)
workflow.add_node("clinical", clinical_node)
workflow.add_node("web", web_node)
workflow.add_node("synthesizer", synthesizer_node)

# Set Entry - start with preflight
workflow.set_entry_point("preflight")

# Add Conditional Edge from Preflight
workflow.add_conditional_edges(
    "preflight",
    route_preflight,
    {
        "planner": "planner",
        "synthesizer": "synthesizer"
    }
)

# Add Conditional Edges (Router)
workflow.add_conditional_edges(
    "planner",
    route_step,
    {
        "iqvia": "iqvia",
        "exim": "exim",
        "patent": "patent",
        "clinical": "clinical",
        "web": "web",
        "synthesizer": "synthesizer"
    }
)

# Add Edges from Workers to Synthesizer
workflow.add_edge("iqvia", "synthesizer")
workflow.add_edge("exim", "synthesizer")
workflow.add_edge("patent", "synthesizer")
workflow.add_edge("clinical", "synthesizer")
workflow.add_edge("web", "synthesizer")

workflow.add_edge("synthesizer", END)

# Compile
app = workflow.compile()

# --- Execution ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("       GLOSER AI - Debug Mode")
    print("="*60)
    while 1:
        user_input = input("\nEnter your query: ")
        
        print("\n" + "-"*60)
        print("[Main] Starting pipeline...")
        print("-"*60)
        
        inputs = {
            "input_query": user_input,
            "results": [],
            "visuals": [],
            "skip_pipeline": False,
            "preflight_response": "",
            "memory_context": "",
            "is_follow_up": False,
            "key_topics": []
        }
        
        result = app.invoke(inputs)
        
        print("\n" + "="*60)
        print("                FINAL RESULT")
        print("="*60)
        print("\n[Final Answer]:")
        print(result['final_answer'])
        
        if result.get('visuals'):
            print(f"\n[Visuals Generated]: {len(result['visuals'])}")
            for i, vis in enumerate(result['visuals'], 1):
                print(f"  {i}. {vis.get('type', 'unknown').upper()}: {vis.get('title', 'Untitled')}")
                if vis.get('type') == 'table':
                    print(f"     Columns: {vis.get('columns', [])}")
                    print(f"     Rows: {len(vis.get('rows', []))}")
                elif vis.get('datasets'):
                    print(f"     Labels: {vis.get('labels', [])[:5]}...")
                    print(f"     Data points: {len(vis.get('datasets', [{}])[0].get('data', []))}")
        else:
            print("\n[Visuals Generated]: None")
        
        print("\n" + "="*60)