"""RAG agent simulator with a MockLLM for lightweight testing."""
from typing import List, Dict, Optional


class MockLLM:
    """Simple heuristic LLM: selects the most-mentioned tool in retrieved docs."""
    def select_tool(self, retrieved_docs: List[Dict], available_tools: List[str]) -> Optional[str]:
        counts = {}
        for doc in retrieved_docs:
            tools = doc.get('tools_mentioned', [])
            for t in tools:
                counts[t] = counts.get(t, 0) + 1
        if not counts:
            return None
        # pick highest count among available_tools
        sorted_tools = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        for tool, _ in sorted_tools:
            if tool in available_tools:
                return tool
        return sorted_tools[0][0]


class RAGAgent:
    def __init__(self, knowledge_base, llm_backbone=None, retriever=None, defense_filter=None, k: int = 5):
        self.kb = knowledge_base
        self.llm = llm_backbone or MockLLM()
        self.retriever = retriever or knowledge_base.retriever
        self.defense = defense_filter
        self.k = k

    def run_task(self, query: str, available_tools: List[str], attack_tool: str = None) -> Dict:
        retrieved = self.kb.retrieve(query, k=self.k)
        attack_retrieved = any((d.get('attack_tool') == attack_tool or d.get('label') == 'adversarial') for d in retrieved)
        filtered = retrieved
        concentration_scores = None
        if self.defense is not None:
            filtered, scores = self._apply_defense(query, retrieved)
            concentration_scores = scores

        selected_tool = self.llm.select_tool(filtered, available_tools)
        attack_selected = (selected_tool == attack_tool)

        return {
            'query': query,
            'retrieved_docs': retrieved,
            'filtered_docs': filtered,
            'selected_tool': selected_tool,
            'attack_tool_retrieved': attack_retrieved,
            'attack_tool_selected': attack_selected,
            'concentration_scores': concentration_scores
        }

    def _apply_defense(self, query: str, retrieved: List[Dict]):
        if hasattr(self.defense, 'filter_with_scores'):
            return self.defense.filter_with_scores(query, retrieved)

        try:
            filtered, scores = self.defense.filter(query, retrieved, return_scores=True)
            return filtered, scores
        except TypeError:
            pass

        try:
            filtered = self.defense.filter(query, retrieved)
        except TypeError:
            filtered = self.defense.filter(retrieved)

        return filtered, None

    def evaluate_dataset(self, dataset: List[Dict], use_defense: bool = True) -> Dict:
        original_defense = self.defense
        if not use_defense:
            self.defense = None
        results = []
        try:
            for item in dataset:
                query = item['query']
                attack_tool = item.get('attack_tool')
                available_tools = item.get('available_tools', [])
                res = self.run_task(query, available_tools, attack_tool=attack_tool)
                results.append(res)
        finally:
            self.defense = original_defense

        total = len(results)
        asr_a = sum(1 for r in results if r['attack_tool_selected']) / max(1, total)
        rr = sum(1 for r in results if r['attack_tool_retrieved']) / max(1, total)
        asr_r = sum(1 for r in results if r['attack_tool_retrieved'] and r['attack_tool_selected']) / max(1, total)
        return {
            'ASR_A': asr_a,
            'ASR_R': asr_r,
            'RR': rr,
            'defense_active': use_defense
        }
