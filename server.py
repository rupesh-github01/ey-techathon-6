from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Lock
import traceback
import uuid

# Import master_agent workflow compiled app
import master_agent

# Import memory manager for conversation context
from memory_manager import (
    add_to_memory,
    get_memory_context,
    get_memory_for_llm,
    clear_memory,
    memory_manager
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (allows Next.js frontend to call backend)
invoke_lock = Lock()


@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "master_agent_api"})


@app.route("/api/master-agent/query", methods=["POST"])  # main endpoint to run entire pipeline
def query_master_agent():
    try:
        body = request.get_json() or {}
        query = body.get("query") or body.get("input_query")
        session_id = body.get("session_id") or str(uuid.uuid4())
        
        print(f"\n[Server] Received query: {query[:80]}..." if query and len(query) > 80 else f"\n[Server] Received query: {query}")
        print(f"[Server] Session ID: {session_id}")
        
        if not query:
            return jsonify({"error": "No query provided. Send JSON with 'query' field."}), 400

        plan_only = body.get("plan_only", False)

        # Get memory context for this session
        memory_context = get_memory_context(session_id, query)
        memory_data = get_memory_for_llm(session_id, query)
        
        if memory_context:
            print(f"[Server] Memory context loaded: {len(memory_context)} chars")
            print(f"[Server] Is follow-up: {memory_data.get('is_follow_up', False)}")
            print(f"[Server] Key topics: {memory_data.get('key_topics', [])}")
        
        # Add user message to memory
        add_to_memory(session_id, "user", query)

        # Prepare initial state with all required fields including memory
        inputs = {
            "input_query": query,
            "results": [],
            "visuals": [],
            "skip_pipeline": False,
            "preflight_response": "",
            "memory_context": memory_context,  # Add memory context
            "is_follow_up": memory_data.get("is_follow_up", False),
            "key_topics": memory_data.get("key_topics", [])
        }

        # If plan_only, call the planner node directly
        if plan_only:
            plan_resp = master_agent.planner_node(inputs)
            return jsonify({"query": query, "plan": plan_resp.get("plan", []), "session_id": session_id})

        # Lock to prevent concurrent LLM calls colliding
        print("[Server] Invoking master agent pipeline...")
        with invoke_lock:
            result = master_agent.app.invoke(inputs)

        # Extract the response components
        visuals = result.get("visuals", [])
        final_answer = result.get("final_answer", "")
        
        # Store assistant response in memory
        add_to_memory(session_id, "assistant", final_answer, has_visuals=len(visuals) > 0)
        
        print(f"[Server] Pipeline complete. Final answer: {len(final_answer)} chars")
        print(f"[Server] Visuals returned: {len(visuals)}")
        for i, v in enumerate(visuals, 1):
            print(f"[Server]   {i}. {v.get('type', 'unknown')}: {v.get('title', 'N/A')}")
        
        # Get session stats for debugging
        session_stats = memory_manager.get_session_stats(session_id)
        print(f"[Server] Session stats: {session_stats}")
        
        response = {
            "query": query,
            "session_id": session_id,
            "result": {
                "final_answer": final_answer,
                "visuals": visuals
            },
            "memory_stats": {
                "total_exchanges": session_stats.get("total_exchanges", 0),
                "key_topics": session_stats.get("key_entities", [])
            }
        }
        print(f"[Server] Sending response with {len(visuals)} visual(s)\n")
        return jsonify(response)
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/api/master-agent/plan", methods=["POST"])  # explicit endpoint to return plan only
def get_plan():
    try:
        body = request.get_json() or {}
        query = body.get("query") or body.get("input_query")
        session_id = body.get("session_id") or str(uuid.uuid4())
        
        if not query:
            return jsonify({"error": "No query provided. Send JSON with 'query' field."}), 400

        # Get memory context
        memory_context = get_memory_context(session_id, query)
        memory_data = get_memory_for_llm(session_id, query)

        inputs = {
            "input_query": query,
            "results": [],
            "visuals": [],
            "skip_pipeline": False,
            "preflight_response": "",
            "memory_context": memory_context,
            "is_follow_up": memory_data.get("is_follow_up", False),
            "key_topics": memory_data.get("key_topics", [])
        }
        plan_resp = master_agent.planner_node(inputs)
        return jsonify({"query": query, "plan": plan_resp.get("plan", []), "session_id": session_id})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


@app.route("/api/master-agent/memory/clear", methods=["POST"])  # Clear session memory
def clear_session_memory():
    try:
        body = request.get_json() or {}
        session_id = body.get("session_id")
        
        if not session_id:
            return jsonify({"error": "No session_id provided."}), 400
        
        clear_memory(session_id)
        print(f"[Server] Cleared memory for session: {session_id}")
        return jsonify({"status": "ok", "message": f"Memory cleared for session {session_id}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/master-agent/memory/stats", methods=["POST"])  # Get session memory stats
def get_session_memory_stats():
    try:
        body = request.get_json() or {}
        session_id = body.get("session_id")
        
        if not session_id:
            return jsonify({"error": "No session_id provided."}), 400
        
        stats = memory_manager.get_session_stats(session_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run simple Flask server for local usage
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
