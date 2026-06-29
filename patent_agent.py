import json
from llm_worker import WorkerAgent


def patent_search(query):
    """Searches Patent data."""
    try:
        with open('patent_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = [item for item in data if any(term.lower() in str(value).lower() for term in query.split() for value in item.values())]

        summary = f"Patent Agent found {len(results)} patents matching '{query}'."
        return {"agent": "patent", "data": results, "summary": summary}
    except Exception as e:
        return {"agent": "patent", "data": [], "summary": f"Error: {str(e)}"}


def patent_worker(query: str = ''):
    worker = WorkerAgent('patent', 'patent_data.json')
    return worker.run_task(query)


if __name__ == '__main__':
    import pprint
    pprint.pprint(patent_worker('CardioFix'))