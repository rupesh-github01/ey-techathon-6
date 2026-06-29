import os
import json
from collections import Counter
from typing import Any, Dict, List


class WorkerAgent:
    """Lightweight worker agent that can use an LLM (if configured)
    to decide, attempt simple solutions, and produce a worker-level summary.

    Behavior:
    - If `openai` is available and `OPENAI_API_KEY` is set, it will call the OpenAI
      ChatCompletion API (gpt-3.5/4 styles) to produce a summary and suggested actions.
    - Otherwise it falls back to a deterministic, local summarizer.
    """

    def __init__(self, name: str, data_path: str):
        self.name = name
        self.data_path = data_path

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Normalize data to a list of records.
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                normalized = []
                for k, v in data.items():
                    if isinstance(v, dict):
                        rec = {'_key': k}
                        rec.update(v)
                    else:
                        rec = {'_key': k, 'value': v}
                    normalized.append(rec)
                return normalized
            return []
        except Exception:
            return []

    def _local_summary(self, data: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        # Basic counts and top values for common keys
        total = len(data)
        matched = []
        if query:
            terms = [t.lower() for t in query.split()]
            for item in data:
                # Deep search: stringify nested structures for matching
                item_text = json.dumps(item, ensure_ascii=False)
                if any(term in item_text.lower() for term in terms):
                    matched.append(item)
        else:
            matched = data

        summary = {
            'agent': self.name,
            'total_records': total,
            'matched_records': len(matched),
            'top_counts': {},
            'examples': matched[:5]
        }

        # gather top values for a few likely keys
        if matched:
            keys = set().union(*(d.keys() for d in matched))
            for key in ('status', 'assignee', 'region', 'phase', 'drug', 'patent_id'):
                if key in keys:
                    counter = Counter([d.get(key) for d in matched if d.get(key) is not None])
                    summary['top_counts'][key] = counter.most_common(5)

        # simple suggested actions
        suggestions = []
        if summary['matched_records'] == 0:
            suggestions.append('No matching records: broaden the query or check data path.')
        else:
            suggestions.append('Inspect top matched examples for anomalies.')
            if 'status' in summary['top_counts'] and any(s[0] in ('Pending', 'Terminated', 'Suspended') for s in summary['top_counts']['status']):
                suggestions.append('Flag records with Pending/Terminated/Suspended status for review.')

        summary['suggestions'] = suggestions
        # If this is the clinical worker, return a structure similar to clinical_trials_worker
        if self.name.lower() == 'clinical':
            def pick(d, keys):
                for k in keys:
                    if k in d and d.get(k) is not None:
                        return d.get(k)
                return None

            active = []
            for m in matched:
                active.append({
                    'NCTId': pick(m, ['NCTId', 'trial_id', 'nctId', '_key']),
                    'BriefTitle': pick(m, ['BriefTitle', 'briefTitle', 'title', 'drug']),
                    'OverallStatus': pick(m, ['OverallStatus', 'status', 'overall_status']),
                    'Phase': pick(m, ['Phase', 'phase']),
                    'LeadSponsorName': pick(m, ['LeadSponsorName', 'lead_sponsor', 'sponsor', 'assignee']),
                    'LocationCountry': pick(m, ['LocationCountry', 'country', 'location']),
                })

            # build sponsor profiles
            sponsors = {}
            for a in active:
                s = a.get('LeadSponsorName') or 'Unknown'
                sponsors.setdefault(s, {'sponsor': s, 'n_trials': 0, 'phases': set(), 'countries': set()})
                sponsors[s]['n_trials'] += 1
                if a.get('Phase'):
                    sponsors[s]['phases'].add(str(a.get('Phase')))
                if a.get('LocationCountry'):
                    for country in str(a.get('LocationCountry')).split(','):
                        sponsors[s]['countries'].add(country.strip())

            sponsor_profiles = []
            for s, vals in sponsors.items():
                sponsor_profiles.append({
                    'sponsor': s,
                    'n_trials': vals['n_trials'],
                    'phases': ', '.join(sorted([p for p in vals['phases'] if p])),
                    'countries': ','.join(sorted([c for c in vals['countries'] if c])),
                })

            # phase distribution
            phase_counts = Counter([a.get('Phase') or 'Unknown' for a in active])
            total_m = sum(phase_counts.values()) or 1
            phase_distribution = []
            for ph, cnt in phase_counts.items():
                phase_distribution.append({'phase': ph, 'n_trials': cnt, 'percent': cnt / total_m * 100})

            return {
                'agent': self.name,
                'total_records': total,
                'matched_records': len(matched),
                'active_trials': active,
                'sponsor_profiles': sponsor_profiles,
                'phase_distribution': phase_distribution,
                'examples': matched[:5],
                'suggestions': suggestions,
            }

        return summary

    def _call_openai(self, prompt: str) -> str:
        try:
            import openai
        except Exception:
            raise RuntimeError('openai package not available')

        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY not set')
        openai.api_key = api_key

        # Use ChatCompletion if available, fallback to Completion
        try:
            resp = openai.ChatCompletion.create(
                model=os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'),
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=500,
                temperature=0.2,
            )
            return resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            raise RuntimeError(f'OpenAI call failed: {e}')

    def _call_ollama(self, prompt: str) -> str:
        """Call a local Ollama server (if available) to generate text using a model like Mistral.

        Expects env vars:
        - OLLAMA_URL (optional, default http://localhost:11434)
        - OLLAMA_MODEL (optional, default 'mistral')
        """
        url_base = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
        model = os.environ.get('OLLAMA_MODEL', 'mistral')

        try:
            import requests
        except Exception:
            raise RuntimeError('requests package required for Ollama calls')

        endpoint = url_base.rstrip('/') + '/api/generate'
        payload = {'model': model, 'prompt': prompt}
        try:
            resp = requests.post(endpoint, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Try common response shapes
            if isinstance(data, dict):
                for key in ('text', 'generated_text', 'result', 'output'):
                    if key in data:
                        return data[key] if isinstance(data[key], str) else json.dumps(data[key])
            return str(data)
        except Exception as e:
            raise RuntimeError(f'Ollama call failed: {e}')

    def run_task(self, query: str = '') -> Dict[str, Any]:
        data = self.load_data()
        # Prepare compact context (first N records) for LLM
        context_sample = data[:10]
        # Special-case clinical worker: ask for structured JSON matching the clinical_trials_worker shape
        if self.name.lower() == 'clinical':
            prompt = (
                "You are a clinical data worker. Given the data sample and the query, "
                "produce a JSON object with the following keys:\n"
                "- active_trials: list of trials with fields [NCTId, BriefTitle, OverallStatus, Phase, LocationCountry, LeadSponsorName]\n"
                "- sponsor_profiles: list of sponsor summaries with keys [sponsor, n_trials, phases, countries]\n"
                "- phase_distribution: list of phase counts with keys [phase, n_trials, percent]\n\n"
            )
            prompt += f"Query.condition: {query}\n"
            prompt += "Data sample (JSON):\n"
        else:
            prompt = f"You are a helpful agent summarizing dataset for agent '{self.name}'.\n"
            prompt += f"Query: {query}\n\n"
            prompt += "Data sample (JSON):\n"
        try:
            prompt += json.dumps(context_sample, indent=2, ensure_ascii=False)
        except Exception:
            prompt += str(context_sample)

        if self.name.lower() == 'clinical':
            prompt += "\n\nReturn only valid JSON. The structure must match the keys and field names exactly. "
            prompt += "If a value is missing, use null or empty list as appropriate. Keep arrays reasonably sized (max 50 items)."
        else:
            prompt += "\n\nProvide a concise summary (3-5 bullets) and 2-3 suggested actions the worker can take."

        # Try LLMs in order: Ollama -> OpenAI -> local
        # Ollama preferred when OLLAMA_URL/OLLAMA_MODEL or LLM_BACKEND=='ollama'
        backend = os.environ.get('LLM_BACKEND', '').lower()
        if backend == 'ollama' or os.environ.get('OLLAMA_URL') or os.environ.get('OLLAMA_MODEL'):
            try:
                llm_response = self._call_ollama(prompt)
                # If clinical worker, try to parse JSON
                if self.name.lower() == 'clinical':
                    try:
                        parsed = json.loads(llm_response)
                        return {'agent': self.name, 'source': 'ollama', 'summary': parsed}
                    except Exception:
                        return {'agent': self.name, 'source': 'ollama', 'summary': llm_response}
                return {'agent': self.name, 'source': 'ollama', 'summary': llm_response}
            except Exception:
                pass

        try:
            llm_response = self._call_openai(prompt)
            if self.name.lower() == 'clinical':
                try:
                    parsed = json.loads(llm_response)
                    return {'agent': self.name, 'source': 'openai', 'summary': parsed}
                except Exception:
                    return {'agent': self.name, 'source': 'openai', 'summary': llm_response}
            return {'agent': self.name, 'source': 'openai', 'summary': llm_response}
        except Exception:
            # fallback to local summary
            local = self._local_summary(data, query)
            return {'agent': self.name, 'source': 'local', 'summary': local}


def generate_with_ollama(prompt: str) -> str:
    """Module-level helper to call Ollama directly from other modules."""
    wa = WorkerAgent('ollama-helper', '')
    return wa._call_ollama(prompt)
