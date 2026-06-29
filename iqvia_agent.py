import json
from typing import List
from llm_worker import WorkerAgent


def _deep_search_in_record(rec: dict, terms: List[str]) -> bool:
    try:
        text = json.dumps(rec, ensure_ascii=False).lower()
    except Exception:
        text = str(rec).lower()
    return any(t in text for t in terms)


def iqvia_search(query):
    """Searches IQVIA market data. Works with dict keyed by area or list of records."""
    try:
        with open('iqvia_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        terms = [t.lower() for t in str(query).split()]
        results = []
        if isinstance(data, list):
            for item in data:
                if _deep_search_in_record(item, terms):
                    results.append(item)
        elif isinstance(data, dict):
            for k, v in data.items():
                rec = {'area': k}
                if isinstance(v, dict):
                    rec.update(v)
                else:
                    rec['value'] = v
                if _deep_search_in_record(rec, terms):
                    results.append(rec)
        else:
            results = []

        summary = f"IQVIA Agent found {len(results)} records related to '{query}'."
        return {"agent": "iqvia", "data": results, "summary": summary}
    except Exception as e:
        return {"agent": "iqvia", "data": [], "summary": f"Error: {str(e)}"}


def iqvia_worker(query: str = ''):
    worker = WorkerAgent('iqvia', 'iqvia_data.json')
    return worker.run_task(query)


if __name__ == '__main__':
    import pprint
    pprint.pprint(iqvia_search('cardio'))