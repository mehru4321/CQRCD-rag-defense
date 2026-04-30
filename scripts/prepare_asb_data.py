"""Convert raw ASB JSONL files into the CQRCD dataset format.

Default behavior matches the implementation plan's 400-scenario setup:
one representative task per ASB agent/domain times that agent's 40 attack tools.
Use ``--all-tasks`` to generate adversarial documents for all 50 ASB tasks.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from config import N_NEGATIVES, RANDOM_SEED, WHITEBOX_OPTIM_STEPS
from modules.dsrm_simulator import DSRMSimulator


AGENT_DOMAIN = {
    'financial_analyst_agent': 'finance',
    'legal_consultant_agent': 'legal',
    'medical_advisor_agent': 'medicine',
    'education_consultant_agent': 'education',
    'psychological_counselor_agent': 'psychology',
    'ecommerce_manager_agent': 'ecommerce',
    'aerospace_engineer_agent': 'aerospace',
    'academic_search_agent': 'academic_search',
    'autonomous_driving_agent': 'autonomous_driving',
    'system_admin_agent': 'system_admin',
}


def read_jsonl(path: Path) -> List[Dict]:
    with path.open('r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(list(rows), handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def normalize_tool_name(tool: Dict, key: str) -> str:
    return str(tool[key]).strip()


def build_tasks(raw_tasks: List[Dict], normal_by_agent: Dict[str, List[Dict]]) -> List[Dict]:
    tasks = []
    for agent_row in raw_tasks:
        agent = agent_row['agent_name']
        domain = AGENT_DOMAIN.get(agent, agent.replace('_agent', ''))
        legitimate_tools = [normalize_tool_name(t, 'Tool Name') for t in normal_by_agent.get(agent, [])]
        for task_idx, query in enumerate(agent_row['tasks'], start=1):
            tasks.append({
                'task_id': f'{domain}_{task_idx:03d}',
                'agent_name': agent,
                'agent_path': agent_row.get('agent_path'),
                'domain': domain,
                'query': query,
                'available_tools': legitimate_tools,
                'is_representative': task_idx == 1,
            })
    return tasks


def build_legitimate_kb(tasks: List[Dict], normal_tools: List[Dict]) -> List[Dict]:
    first_task_by_agent = {}
    for task in tasks:
        first_task_by_agent.setdefault(task['agent_name'], task)

    docs = []
    counters = defaultdict(int)
    for tool in normal_tools:
        agent = tool['Corresponding Agent']
        task = first_task_by_agent[agent]
        domain = task['domain']
        counters[domain] += 1
        tool_name = normalize_tool_name(tool, 'Tool Name')
        text = (
            f"Historical experience for {domain}: A prior user asked, \"{task['query']}\" "
            f"The agent selected {tool_name}. Tool description: {tool['Description']} "
            f"Outcome: {tool['Expected Achievements']}"
        )
        docs.append({
            'id': f'doc_{domain}_{counters[domain]:03d}',
            'domain': domain,
            'agent_name': agent,
            'text': text,
            'label': 'legitimate',
            'tools_mentioned': [tool_name],
            'source_query': task['query'],
            'tool_description': tool['Description'],
            'expected_achievements': tool['Expected Achievements'],
        })
    return docs


def build_adversarial_docs(
    tasks: List[Dict],
    attack_tools: List[Dict],
    normal_by_agent: Dict[str, List[Dict]],
    all_tasks: bool,
) -> tuple[List[Dict], List[Dict]]:
    tasks_by_agent = defaultdict(list)
    for task in tasks:
        if all_tasks or task['is_representative']:
            tasks_by_agent[task['agent_name']].append(task)

    attacks_by_agent = defaultdict(list)
    for attack in attack_tools:
        attacks_by_agent[attack['Corresponding Agent']].append(attack)

    blackbox = []
    whitebox = []
    bb_sim = DSRMSimulator(seed=RANDOM_SEED)
    wb_sim = DSRMSimulator(seed=RANDOM_SEED + 1)

    for agent in sorted(tasks_by_agent):
        legitimate_tools = [normalize_tool_name(t, 'Tool Name') for t in normal_by_agent.get(agent, [])]
        for task in tasks_by_agent[agent]:
            for attack_idx, attack in enumerate(attacks_by_agent.get(agent, []), start=1):
                attack_tool = normalize_tool_name(attack, 'Attacker Tool')
                common_metadata = {
                    'domain': task['domain'],
                    'agent_name': agent,
                    'task_id': task['task_id'],
                    'attack_type': attack.get('Attack Type'),
                    'attack_goal': attack.get('Attack goal'),
                    'attack_instruction': attack.get('Attacker Instruction'),
                    'aggressive': str(attack.get('Aggressive', '')).lower() == 'true',
                    'attack_tool_description': attack.get('Description'),
                    'scenario_id': f"{task['task_id']}_attack_{attack_idx:03d}",
                }

                bb_doc = bb_sim.generate_blackbox(
                    query=task['query'],
                    attack_tool=attack_tool,
                    attack_instruction=attack['Attacker Instruction'],
                    legitimate_tools=legitimate_tools,
                )
                bb_doc.update(common_metadata)
                blackbox.append(bb_doc)

                wb_doc = wb_sim.generate_whitebox(
                    query=task['query'],
                    attack_tool=attack_tool,
                    attack_instruction=attack['Attacker Instruction'],
                    legitimate_tools=legitimate_tools,
                    n_steps=WHITEBOX_OPTIM_STEPS,
                    n_negatives=N_NEGATIVES,
                )
                wb_doc.update(common_metadata)
                whitebox.append(wb_doc)

    return blackbox, whitebox


def validate_counts(tasks: List[Dict], legit: List[Dict], bb: List[Dict], wb: List[Dict], all_tasks: bool) -> None:
    expected_adv = len(tasks) * 40 if all_tasks else 400
    if len({task['agent_name'] for task in tasks}) != 10:
        raise ValueError('Expected 10 ASB agents/domains')
    if len(legit) != 20:
        raise ValueError(f'Expected 20 legitimate KB docs, found {len(legit)}')
    if len(bb) != expected_adv or len(wb) != expected_adv:
        raise ValueError(f'Expected {expected_adv} black-box and white-box docs, found {len(bb)} / {len(wb)}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare CQRCD JSON data from raw ASB JSONL files.')
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw_asb'))
    parser.add_argument('--out-dir', type=Path, default=Path('data'))
    parser.add_argument('--all-tasks', action='store_true', help='Generate adversarial docs for all 50 ASB tasks.')
    args = parser.parse_args()

    raw_tasks = read_jsonl(args.raw_dir / 'agent_task.jsonl')
    normal_tools = read_jsonl(args.raw_dir / 'all_normal_tools.jsonl')
    attack_tools = read_jsonl(args.raw_dir / 'all_attack_tools.jsonl')

    normal_by_agent = defaultdict(list)
    for tool in normal_tools:
        normal_by_agent[tool['Corresponding Agent']].append(tool)

    tasks = build_tasks(raw_tasks, normal_by_agent)
    legitimate_kb = build_legitimate_kb(tasks, normal_tools)
    adversarial_bb, adversarial_wb = build_adversarial_docs(
        tasks,
        attack_tools,
        normal_by_agent,
        all_tasks=args.all_tasks,
    )

    validate_counts(tasks, legitimate_kb, adversarial_bb, adversarial_wb, args.all_tasks)

    write_json(args.out_dir / 'asb_tasks.json', tasks)
    write_json(args.out_dir / 'legitimate_kb.json', legitimate_kb)
    write_json(args.out_dir / 'adversarial_bb.json', adversarial_bb)
    write_json(args.out_dir / 'adversarial_wb.json', adversarial_wb)

    print('Prepared CQRCD data from ASB:')
    print(f'  tasks: {len(tasks)} -> {args.out_dir / "asb_tasks.json"}')
    print(f'  legitimate KB docs: {len(legitimate_kb)} -> {args.out_dir / "legitimate_kb.json"}')
    print(f'  black-box adversarial docs: {len(adversarial_bb)} -> {args.out_dir / "adversarial_bb.json"}')
    print(f'  white-box adversarial docs: {len(adversarial_wb)} -> {args.out_dir / "adversarial_wb.json"}')


if __name__ == '__main__':
    main()
