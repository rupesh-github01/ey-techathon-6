from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# --- 1. DEFINE THE STATE ---
# This is the "Local Memory" for this specific agent.
class IqviaState(TypedDict):
    query: str              # Input from Master Agent
    messages: List[any]     # Internal thought process (chat history)
    data: dict              # The structured data retrieved
    final_output: str       # The formatted summary for the Master Agent

# --- 2. DEFINE THE MOCK TOOL ---
# This simulates the API. In the real hackathon, this queries your CSV/JSON.
@tool
def fetch_iqvia_data(therapy_area: str, metric: str = "all"):
    """
    Fetches market data for a specific therapy area.
    Args:
        therapy_area: e.g., 'Oncology', 'Diabetes', 'Cardiovascular'
        metric: 'sales', 'cagr', 'competitors', or 'all'
    """
    # MOCK DATABASE
    mock_db = {
    "oncology": {
        "market_size_usd": "185 Billion",
        "cagr_percent": 12.5,
        "volume_units": "950 Million Doses",
        "competitors": ["Roche", "Merck", "Bristol Myers Squibb", "AstraZeneca"],
        "competition_level": "High",
        "key_trend": "Shift towards CAR-T therapies and personalized medicine."
    },
    "diabetes": {
        "market_size_usd": "90 Billion",
        "cagr_percent": 6.1,
        "volume_units": "2.1 Billion Units",
        "competitors": ["Novo Nordisk", "Eli Lilly", "Sanofi"],
        "competition_level": "Very High",
        "key_trend": "Rapid adoption of GLP-1 agonists (e.g., Ozempic, Mounjaro)."
    },
    "respiratory": {
        "market_size_usd": "45 Billion",
        "cagr_percent": 3.4,
        "volume_units": "1.8 Billion Inhalers",
        "competitors": ["GSK", "AstraZeneca", "Boehringer Ingelheim"],
        "competition_level": "Moderate",
        "key_trend": "Generic erosion in asthma; growth in COPD biologics."
    },
    "immunology": {
        "market_size_usd": "110 Billion",
        "cagr_percent": 9.8,
        "volume_units": "600 Million Doses",
        "competitors": ["AbbVie", "J&J", "Amgen"],
        "competition_level": "High",
        "key_trend": "Biosimilar competition entering for Humira; shift to JAK inhibitors."
    },
    "cardiovascular": {
        "market_size_usd": "60 Billion",
        "cagr_percent": 2.5,
        "volume_units": "3.5 Billion Tablets",
        "competitors": ["Pfizer", "Bayer", "Novartis"],
        "competition_level": "Saturated",
        "key_trend": "High generic penetration; growth in heart failure novel therapies."
    },
    "neurology": {
        "market_size_usd": "55 Billion",
        "cagr_percent": 8.2,
        "volume_units": "400 Million Doses",
        "competitors": ["Biogen", "Roche", "Eisai"],
        "competition_level": "Moderate",
        "key_trend": "Emerging treatments for Alzheimer's showing promise but high cost."
    },
    "rare_diseases": {
        "market_size_usd": "25 Billion",
        "cagr_percent": 14.0,
        "volume_units": "5 Million Doses",
        "competitors": ["Takeda", "Sanofi Genzyme", "Vertex"],
        "competition_level": "Low",
        "key_trend": "High unmet need; heavily incentivized by Orphan Drug designations."
    }

    }
    
    # Fuzzy matching logic (simulated)
    key = therapy_area.lower()
    if key in mock_db:
        return mock_db[key]
    return {"error": "Therapy area not found. Try: Oncology, Diabetes, Cardiovascular"}

# --- 3. DEFINE THE NODES ---

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([fetch_iqvia_data])

def reasoner_node(state: IqviaState):
    """
    The Brain: Decides if we need to call a tool or if we are done.
    """
    messages = state.get("messages", [])
    if not messages:
        # First turn: Add system instructions + user query
        messages = [
            SystemMessage(content="You are a Market Data Analyst. Retrieve structured data."),
            HumanMessage(content=state["query"])
        ]
    
    # The LLM sees the tool definition and decides what to do
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: IqviaState):
    """
    The Hands: Executes the tool if the Reasoner requested it.
    """
    last_message = state["messages"][-1]
    
    # If no tool call, do nothing
    if not last_message.tool_calls:
        return {}

    # Execute the tool
    tool_call = last_message.tool_calls[0]
    tool_output = fetch_iqvia_data.invoke(tool_call)
    
    # Save the raw data specifically to the state
    return {
        "messages": [tool_output], # Add tool output to history
        "data": tool_output        # Store explicitly for easy access
    }

def formatter_node(state: IqviaState):
    """
    The Reporter: Formats the raw data into a clean string for the Master Agent.
    """
    raw_data = state.get("data")
    if not raw_data or "error" in raw_data:
        return {"final_output": "Data not available."}
    
    # Create a nice summary string
    summary = (
        f"MARKET DATA FOUND:\n"
        f"- Size: {raw_data.get('size')}\n"
        f"- Growth (CAGR): {raw_data.get('cagr')}\n"
        f"- Key Players: {', '.join(raw_data.get('competitors', []))}"
    )
    return {"final_output": summary}

# --- 4. BUILD THE GRAPH ---

workflow = StateGraph(IqviaState)

workflow.add_node("reasoner", reasoner_node)
workflow.add_node("executor", tool_node)
workflow.add_node("formatter", formatter_node)

workflow.set_entry_point("reasoner")

# Conditional Logic: Did the LLM ask for a tool?
def should_continue(state: IqviaState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "executor" # Go to tool execution
    return "formatter"    # We have the data (or failed), go to formatting

workflow.add_conditional_edges("reasoner", should_continue)
workflow.add_edge("executor", "reasoner") # Loop back to see if more data is needed
workflow.add_edge("formatter", END)

# COMPILE - This object is what you import in the Master Agent
iqvia_agent_app = workflow.compile()
