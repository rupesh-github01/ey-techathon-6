import json
import os
from typing import List, Any, Dict
from urllib.parse import quote_plus

from llm_worker import WorkerAgent

# ClinicalTrials.gov API base URL (v2)
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def fetch_trials_v2(condition, country=None, status="RECRUITING", max_records=50):
    """Fetch studies from ClinicalTrials.gov v2 and return a list of simplified records.

    This implementation avoids a hard dependency on pandas by returning a list
    of dicts (rows). Each row contains keys matching the fields requested.
    """
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"requests required to call ClinicalTrials.gov: {e}")

    params = {
        "query.cond": condition,
        "fields": "NCTId,BriefTitle,OverallStatus,Phase,LocationCountry,LeadSponsorName",
        "pageSize": max_records,
        "countTotal": "true",
    }
    if country:
        params["query.locn"] = country
    if status:
        params["filter.overallStatus"] = status.upper()

    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    studies = data.get("studies", []) if isinstance(data, dict) else []
    rows = []
    for s in studies:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        loc_mod = proto.get("contactsLocationsModule", {})

        locs = loc_mod.get("locations", []) if isinstance(loc_mod, dict) else []
        countries = sorted({loc.get("country") for loc in locs if loc.get("country")}) if locs else []

        rows.append({
            "NCTId": ident.get("nctId"),
            "BriefTitle": ident.get("briefTitle"),
            "OverallStatus": status_mod.get("overallStatus"),
            "Phase": status_mod.get("phase"),
            "LeadSponsorName": (sponsor_mod.get("leadSponsor") or {}).get("name") if isinstance(sponsor_mod, dict) else None,
            "LocationCountry": ",".join(countries) if countries else None,
        })

    return rows


def build_sponsor_profiles_from_rows(rows):
    if not rows:
        return []
    sponsors = {}
    for r in rows:
        s = r.get("LeadSponsorName") or "Unknown"
        sponsors.setdefault(s, {"sponsor": s, "n_trials": 0, "phases": set(), "countries": set()})
        sponsors[s]["n_trials"] += 1
        ph = r.get("Phase")
        if ph:
            sponsors[s]["phases"].add(str(ph))
        loc = r.get("LocationCountry")
        if loc:
            for c in str(loc).split(","):
                if c:
                    sponsors[s]["countries"].add(c.strip())

    out = []
    for s, vals in sponsors.items():
        out.append({
            "sponsor": s,
            "n_trials": vals["n_trials"],
            "phases": ", ".join(sorted([p for p in vals["phases"] if p])),
            "countries": ",".join(sorted([c for c in vals["countries"] if c])),
        })
    return out


def build_phase_distribution_from_rows(rows):
    if not rows:
        return []
    counts = {}
    for r in rows:
        ph = r.get("Phase") or "Unknown"
        counts[ph] = counts.get(ph, 0) + 1
    total = sum(counts.values()) or 1
    out = []
    for ph, cnt in counts.items():
        out.append({"phase": ph, "n_trials": cnt, "percent": cnt / total * 100})
    return out


def clinical_trials_worker(payload: dict):
    """Payload keys: condition, country, status, max_records

    Returns the structured dict with active_trials, sponsor_profiles, phase_distribution
    """
    condition = payload.get("condition")
    if not condition:
        raise ValueError("payload must include 'condition'")
    country = payload.get("country")
    status = payload.get("status", "RECRUITING")
    max_rec = int(payload.get("max_records", 50) or 50)

    rows = fetch_trials_v2(condition, country=country, status=status, max_records=max_rec)

    result = {
        "active_trials": rows,
        "sponsor_profiles": build_sponsor_profiles_from_rows(rows),
        "phase_distribution": build_phase_distribution_from_rows(rows),
    }

    return result


def clinical_search(query: str, use_api: bool = True, max_results: int = 20) -> Dict[str, Any]:
    """Search clinical trials.

    By default this will try ClinicalTrials.gov API (`BASE_URL`) and fall back
    to the local `clinical_data.json` if the API is unavailable or returns
    unexpected data.
    """
    # Try to parse a human query like "type 2 diabetes in Brazil" into condition+country
    def _parse_condition_country(q: str):
        import re
        if not q:
            return q, None
        s = q.strip()
        # try to find " in <Country>" pattern
        m = re.search(r"\bin\s+([A-Za-z\u00C0-\u017F \-]+)", s, flags=re.IGNORECASE)
        country = None
        if m:
            country = m.group(1).strip().strip(',')
            # remove the matched " in <country>" from the condition text
            s = (s[:m.start()] + s[m.end():]).strip()

        # remove common verbs/phrases at the start
        s = re.sub(r"(?i)^(for|about|tell me about|show|summarize|give me|what are|list)\b", "", s).strip()
        # remove trailing command-like phrases (e.g., ', summarize the clinical trials')
        s = re.sub(r"(?i),?\s*(summarize.*|please.*|show.*|tell me.*|give me.*|list.*)$", "", s).strip()
        # remove trailing phrases like 'clinical trials' or 'trials'
        s = re.sub(r"(?i)\bclinical trials\b|\btrials\b", "", s).strip(' ,.')
        return s or q, country

    condition, parsed_country = _parse_condition_country(str(query))
    country = parsed_country

    if use_api:
        try:
            rows = fetch_trials_v2(condition, country=country, status=None, max_records=max_results)
            summary = f"Clinical Agent (API) returned {len(rows)} records for '{condition}' (country={country})."
            return {"agent": "clinical", "data": rows, "summary": summary}
        except Exception as e:
            api_err = str(e)
    else:
        api_err = "API disabled"

    # Local fallback
    try:
        with open('clinical_data.json', 'r', encoding='utf-8') as f:
            local = json.load(f)

        # normalize list or dict
        records: List[Dict[str, Any]] = []
        if isinstance(local, list):
            records = local
        elif isinstance(local, dict):
            for k, v in local.items():
                rec = {"_key": k} if not isinstance(v, dict) else {"_key": k, **v}
                records.append(rec)

        terms = [t.lower() for t in str(query).split()]
        matched = []
        for r in records:
            try:
                text = json.dumps(r, ensure_ascii=False).lower()
            except Exception:
                text = str(r).lower()
            if any(t in text for t in terms):
                matched.append(r)

        summary = f"Clinical Agent (local) found {len(matched)} trials for '{condition or query}'. (API error: {api_err})"
        return {"agent": "clinical", "data": matched, "summary": summary}
    except Exception as e:
        return {"agent": "clinical", "data": [], "summary": f"Error: {str(e)}; API error: {api_err}"}


def clinical_worker(query: object = ''):
    """Wrapper used by other agents/tools.

    If `query` is a dict it's treated as a payload for `clinical_trials_worker`.
    Otherwise we fallback to the LLM/local WorkerAgent behavior.
    """
    # payload-style call
    if isinstance(query, dict):
        return clinical_trials_worker(query)

    # string query -> use WorkerAgent summarizer/fallback
    worker = WorkerAgent('clinical', 'clinical_data.json')
    return worker.run_task(str(query or ''))


if __name__ == '__main__':
    # Quick development run: try API then local fallback
    import pprint

    q = input("Enter clinical query (e.g., 'Phase 3 CardioFix'): ")
    res = clinical_search(q)
    pprint.pprint(res)