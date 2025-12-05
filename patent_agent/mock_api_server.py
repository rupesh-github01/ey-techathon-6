"""
Unified Mock API Server for EY Techathon 6.0 - Pharmaceutical Innovation Agent
Includes: USPTO Patents, Clinical Trials, IQVIA Market Data, EXIM Trade Data, Internal Docs
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Get the data directory path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Load all data on startup
def load_json(filename):
    # Try data dir first, then fall back to repo root
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        # fallback to same directory as the script
        filepath_alt = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath_alt):
            filepath = filepath_alt
        else:
            print(f"Warning: {filename} not found")
            return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: {filename} not loadable: {e}")
        return []

PATENTS = load_json('patents.json')
CLINICAL_TRIALS = load_json('clinical_trials.json')
IQVIA_DATA = load_json('iqvia_market.json')
EXIM_DATA = load_json('exim_trade.json')
INTERNAL_DOCS = load_json('internal_docs.json')

# =============================================================================
# USPTO PATENT API ENDPOINTS
# =============================================================================

@app.route('/api/patents/search', methods=['GET'])
def search_patents():
    """
    Search patents by keyword in title, abstract, or claims.
    Query Params: q (search query), molecule, indication
    """
    query = request.args.get('q', '').lower()
    molecule = request.args.get('molecule', '').lower()
    indication = request.args.get('indication', '').lower()
    
    results = []
    # If query matches a patent_id exactly, return that single patent
    if query and any(query == p['patent_id'].lower() for p in PATENTS):
        patent = next((p for p in PATENTS if query == p['patent_id'].lower()), None)
        return jsonify({
            "total_results": 1,
            "patents": [patent]
        })

    for patent in PATENTS:
        match = False
        if query:
            searchable = f"{patent['title']} {patent['abstract']} {patent['claims']}".lower()
            if query in searchable:
                match = True
        if molecule and molecule in patent.get('molecule', '').lower():
            match = True
        if indication and indication in patent.get('indication', '').lower():
            match = True
        if not query and not molecule and not indication:
            match = True
        
        if match:
            results.append(patent)
    
    return jsonify({
        "total_results": len(results),
        "patents": results
    })

@app.route('/api/patents/status', methods=['GET'])
def check_patent_status():
    """
    Check legal status of a specific patent.
    Query Param: patent_id
    """
    p_id = request.args.get('patent_id')
    patent = next((p for p in PATENTS if p['patent_id'] == p_id), None)
    
    if not patent:
        return jsonify({"error": "Patent not found"}), 404
    
    today = datetime.now().strftime('%Y-%m-%d')
    is_expired = patent['expiry_date'] < today or patent['status'] == 'Expired'
    
    return jsonify({
        "patent_id": patent['patent_id'],
        "title": patent['title'],
        "assignee": patent['assignee'],
        "status": patent['status'],
        "filing_date": patent.get('filing_date', 'N/A'),
        "expiry_date": patent['expiry_date'],
        "is_expired": is_expired,
        "is_generic_opportunity": is_expired,
        "cpc_codes": patent.get('cpc_codes', [])
    })

@app.route('/api/patents/landscape', methods=['GET'])
def patent_landscape():
    """
    Get patent landscape summary for a molecule.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    
    relevant_patents = [p for p in PATENTS if molecule in p.get('molecule', '').lower()]
    
    status_summary = {"Active": 0, "Pending": 0, "Expired": 0}
    indication_breakdown = {}
    
    for p in relevant_patents:
        status_summary[p['status']] = status_summary.get(p['status'], 0) + 1
        ind = p.get('indication', 'Unknown')
        indication_breakdown[ind] = indication_breakdown.get(ind, 0) + 1
    
    return jsonify({
        "molecule": molecule,
        "total_patents": len(relevant_patents),
        "status_summary": status_summary,
        "indication_breakdown": indication_breakdown,
        "patents": relevant_patents,
        "fto_assessment": "Favorable" if status_summary['Expired'] > status_summary['Active'] else "Requires Navigation"
    })


@app.route('/api/patents/filings', methods=['GET'])
def patent_filings():
    """
    Returns filings by year for a given molecule.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    relevant_patents = [p for p in PATENTS if molecule in p.get('molecule', '').lower()]
    filings = {}
    for p in relevant_patents:
        filing_date = p.get('filing_date', '')
        if filing_date:
            year = filing_date.split('-')[0]
            filings[year] = filings.get(year, 0) + 1

    return jsonify({
        "molecule": molecule,
        "filings_by_year": filings,
        "total_patents": len(relevant_patents)
    })


@app.route('/api/patents/trends', methods=['GET'])
def patent_trends():
    """
    Returns simple trends for a molecule: top CPC codes and most common indications.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    relevant_patents = [p for p in PATENTS if molecule in p.get('molecule', '').lower()]

    cpc_counts = {}
    indication_counts = {}
    for p in relevant_patents:
        for c in p.get('cpc_codes', []):
            cpc_counts[c] = cpc_counts.get(c, 0) + 1
        ind = p.get('indication', 'Unknown')
        indication_counts[ind] = indication_counts.get(ind, 0) + 1

    # sort
    top_cpcs = sorted(cpc_counts.items(), key=lambda x: x[1], reverse=True)
    top_indications = sorted(indication_counts.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "molecule": molecule,
        "top_cpcs": top_cpcs,
        "top_indications": top_indications,
        "total_patents": len(relevant_patents)
    })

# =============================================================================
# CLINICAL TRIALS API ENDPOINTS
# =============================================================================

@app.route('/api/trials/search', methods=['GET'])
def search_trials():
    """
    Search clinical trials by molecule, condition, or status.
    Query Params: molecule, condition, status, phase
    """
    molecule = request.args.get('molecule', '').lower()
    condition = request.args.get('condition', '').lower()
    status = request.args.get('status', '').lower()
    phase = request.args.get('phase', '').lower()
    
    results = []
    for trial in CLINICAL_TRIALS:
        match = True
        
        if molecule and molecule not in trial.get('molecule', '').lower():
            match = False
        if condition:
            conditions_str = ' '.join(trial.get('conditions', [])).lower()
            if condition not in conditions_str:
                match = False
        if status and status not in trial.get('status', '').lower():
            match = False
        if phase and phase not in trial.get('phase', '').lower():
            match = False
        
        if match:
            results.append(trial)
    
    return jsonify({
        "total_results": len(results),
        "trials": results
    })

@app.route('/api/trials/<nct_id>', methods=['GET'])
def get_trial(nct_id):
    """Get details for a specific trial by NCT ID."""
    trial = next((t for t in CLINICAL_TRIALS if t['nct_id'] == nct_id), None)
    
    if not trial:
        return jsonify({"error": "Trial not found"}), 404
    
    return jsonify(trial)

@app.route('/api/trials/pipeline', methods=['GET'])
def trial_pipeline():
    """
    Get clinical trial pipeline summary for a molecule.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    
    relevant_trials = [t for t in CLINICAL_TRIALS if molecule in t.get('molecule', '').lower()]
    
    phase_summary = {}
    status_summary = {}
    indication_summary = {}
    
    for t in relevant_trials:
        phase = t.get('phase', 'Unknown')
        phase_summary[phase] = phase_summary.get(phase, 0) + 1
        
        status = t.get('status', 'Unknown')
        status_summary[status] = status_summary.get(status, 0) + 1
        
        for cond in t.get('conditions', []):
            indication_summary[cond] = indication_summary.get(cond, 0) + 1
    
    return jsonify({
        "molecule": molecule,
        "total_trials": len(relevant_trials),
        "phase_distribution": phase_summary,
        "status_distribution": status_summary,
        "indications_being_studied": indication_summary,
        "trials": relevant_trials
    })

# =============================================================================
# IQVIA MARKET DATA API ENDPOINTS
# =============================================================================

@app.route('/api/market/overview', methods=['GET'])
def market_overview():
    """
    Get market overview for a molecule or therapeutic area.
    Query Params: molecule, therapy_area
    """
    molecule = request.args.get('molecule', '').lower()
    
    if molecule == 'metformin':
        data = IQVIA_DATA.get('metformin', {})
        return jsonify({
            "molecule": "Metformin",
            "therapeutic_area": data.get('therapeutic_area'),
            "global_market": data.get('global_market'),
            "regional_breakdown": data.get('regional_breakdown'),
            "formulation_breakdown": data.get('formulation_breakdown'),
            "future_projections": data.get('future_projections')
        })
    
    return jsonify({"error": "Molecule data not available"}), 404

@app.route('/api/market/opportunities', methods=['GET'])
def market_opportunities():
    """
    Get unmet needs and opportunities for a molecule.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    
    if molecule == 'metformin':
        data = IQVIA_DATA.get('metformin', {})
        return jsonify({
            "molecule": "Metformin",
            "unmet_needs": data.get('unmet_needs', []),
            "opportunities": data.get('opportunities', []),
            "competitive_landscape": data.get('alzheimer_therapeutics', {})
        })
    
    return jsonify({"error": "Molecule data not available"}), 404

@app.route('/api/market/competition', methods=['GET'])
def market_competition():
    """
    Get competitive landscape for a therapeutic area.
    Query Param: therapy_area
    """
    therapy_area = request.args.get('therapy_area', '').lower()
    
    if 'alzheimer' in therapy_area:
        data = IQVIA_DATA.get('alzheimer_therapeutics', {})
        return jsonify(data)
    
    return jsonify({"error": "Therapy area data not available"}), 404

# =============================================================================
# EXIM TRADE DATA API ENDPOINTS
# =============================================================================

@app.route('/api/exim/trade', methods=['GET'])
def exim_trade():
    """
    Get export-import trade data for a molecule/API.
    Query Params: molecule, product_type (api/finished)
    """
    molecule = request.args.get('molecule', '').lower()
    product_type = request.args.get('product_type', 'api').lower()
    
    if 'metformin' in molecule:
        if product_type == 'api':
            data = EXIM_DATA.get('metformin_hcl_api', {})
        else:
            data = EXIM_DATA.get('metformin_finished_dosage', {})
        return jsonify(data)
    
    return jsonify({"error": "Trade data not available"}), 404

@app.route('/api/exim/suppliers', methods=['GET'])
def exim_suppliers():
    """
    Get major suppliers/exporters for a molecule.
    Query Param: molecule
    """
    molecule = request.args.get('molecule', '').lower()
    
    if 'metformin' in molecule:
        data = EXIM_DATA.get('metformin_hcl_api', {})
        return jsonify({
            "molecule": "Metformin HCl",
            "major_exporters": data.get('major_exporters', []),
            "supply_chain_insights": data.get('supply_chain_insights', {}),
            "price_trends": data.get('price_trends', {})
        })
    
    return jsonify({"error": "Supplier data not available"}), 404

# =============================================================================
# INTERNAL DOCS API ENDPOINTS
# =============================================================================

@app.route('/api/internal/search', methods=['GET'])
def search_internal():
    """
    Search internal documents.
    Query Params: q (search query), doc_type
    """
    query = request.args.get('q', '').lower()
    doc_type = request.args.get('doc_type', '').lower()
    
    results = []
    for doc in INTERNAL_DOCS:
        match = False
        
        if query:
            searchable = f"{doc['title']} {doc['content']}".lower()
            if query in searchable:
                match = True
        
        if doc_type and doc_type in doc.get('doc_type', '').lower():
            match = True
        
        if not query and not doc_type:
            match = True
        
        if match:
            # Return summary without full content
            results.append({
                "doc_id": doc['doc_id'],
                "title": doc['title'],
                "doc_type": doc['doc_type'],
                "date": doc['date'],
                "key_insights": doc.get('key_insights', [])
            })
    
    return jsonify({
        "total_results": len(results),
        "documents": results
    })

@app.route('/api/internal/<doc_id>', methods=['GET'])
def get_internal_doc(doc_id):
    """Get full content of an internal document."""
    doc = next((d for d in INTERNAL_DOCS if d['doc_id'] == doc_id), None)
    
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    
    return jsonify(doc)

# =============================================================================
# WEB SEARCH SIMULATION API
# =============================================================================

@app.route('/api/web/search', methods=['GET'])
def web_search():
    """
    Simulated web search for scientific publications, news, guidelines.
    Query Param: q
    """
    query = request.args.get('q', '').lower()
    
    # Mock web search results
    mock_results = {
        "metformin alzheimer": [
            {
                "title": "Metformin and Cognitive Function: A Systematic Review",
                "source": "PubMed - Lancet Neurology 2024",
                "url": "https://pubmed.ncbi.nlm.nih.gov/example1",
                "snippet": "Meta-analysis of 15 studies shows 23% reduction in Alzheimer's risk among metformin users..."
            },
            {
                "title": "FDA Grants Fast Track for Metformin in Alzheimer's",
                "source": "Reuters Health",
                "url": "https://reuters.com/health/example",
                "snippet": "The FDA has granted fast track designation to metformin for prevention of Alzheimer's disease..."
            }
        ],
        "metformin cancer": [
            {
                "title": "AMPK Activation by Metformin Inhibits Tumor Growth",
                "source": "Nature Medicine",
                "url": "https://nature.com/articles/example",
                "snippet": "Study demonstrates metformin's anti-cancer mechanism through AMPK-mediated mTOR inhibition..."
            }
        ],
        "metformin side effects": [
            {
                "title": "Managing Metformin-Associated GI Side Effects",
                "source": "American Diabetes Association",
                "url": "https://diabetes.org/example",
                "snippet": "Extended-release formulations and gradual dose titration can minimize GI adverse effects..."
            }
        ]
    }
    
    # Find matching results
    results = []
    for key, value in mock_results.items():
        if any(word in query for word in key.split()):
            results.extend(value)
    
    if not results:
        results = [{
            "title": f"Search results for: {query}",
            "source": "Web Search",
            "url": "https://example.com",
            "snippet": f"General information about {query} from various sources..."
        }]
    
    return jsonify({
        "query": query,
        "total_results": len(results),
        "results": results
    })

# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "patents": "/api/patents/search, /api/patents/status, /api/patents/landscape",
            "trials": "/api/trials/search, /api/trials/pipeline",
            "market": "/api/market/overview, /api/market/opportunities",
            "exim": "/api/exim/trade, /api/exim/suppliers",
            "internal": "/api/internal/search",
            "web": "/api/web/search"
        }
    })

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Pharma Innovation Agent - Mock API Server")
    print("=" * 60)
    print("Available endpoints:")
    print("  - Patents:   /api/patents/search, /api/patents/status, /api/patents/landscape")
    print("  - Trials:    /api/trials/search, /api/trials/pipeline")
    print("  - Market:    /api/market/overview, /api/market/opportunities")
    print("  - EXIM:      /api/exim/trade, /api/exim/suppliers")
    print("  - Internal:  /api/internal/search")
    print("  - Web:       /api/web/search")
    print("  - Health:    /api/health")
    print("=" * 60)
    app.run(port=5000, debug=True)
