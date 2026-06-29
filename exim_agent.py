import json
from typing import List
from llm_worker import WorkerAgent


def _deep_search_in_record(rec: dict, terms: List[str]) -> bool:
    try:
        text = json.dumps(rec, ensure_ascii=False).lower()
    except Exception:
        text = str(rec).lower()
    return any(t in text for t in terms)


def exim_search(query):
    """Searches Export/Import data. Works with dict (hs_code keyed) or list shapes."""
    try:
        with open('exim_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        terms = [t.lower() for t in str(query).split()]

        results = []
        if isinstance(data, list):
            for item in data:
                if _deep_search_in_record(item, terms):
                    results.append(item)
        elif isinstance(data, dict):
            # normalize: include hs_code in results
            for k, v in data.items():
                rec = {'hs_code': k}
                if isinstance(v, dict):
                    rec.update(v)
                else:
                    rec['value'] = v
                if _deep_search_in_record(rec, terms):
                    results.append(rec)
        else:
            results = []

        summary = f"Exim Agent found {len(results)} records for '{query}'."
        return {"agent": "exim", "data": results, "summary": summary}
    except Exception as e:
        return {"agent": "exim", "data": [], "summary": f"Error: {str(e)}"}


def exim_worker(query: str = ''):
    worker = WorkerAgent('exim', 'exim_data.json')
    return worker.run_task(query)


if __name__ == '__main__':
    import pprint
    pprint.pprint(exim_search('Paracetamol'))