"""Compare answer row multisets, independent of ordering and provenance IDs."""
import json
from collections import defaultdict, deque


def compare_answers(selected, reference):
    """Preserve duplicate counts and compare complete, typed result values."""
    available = defaultdict(deque)
    for index, item in enumerate(reference):
        available[json.dumps(item['result'], sort_keys=True, ensure_ascii=False)].append(index)
    selected_status = []
    reference_status = ['missing'] * len(reference)
    matched = 0
    for item in selected:
        key = json.dumps(item['result'], sort_keys=True, ensure_ascii=False)
        if available[key]:
            reference_status[available[key].popleft()] = 'shared'
            selected_status.append('shared')
            matched += 1
        else:
            selected_status.append('extra')
    return {
        'matched_count': matched,
        'missing_count': len(reference) - matched,
        'extra_count': len(selected) - matched,
        'answers_equal': matched == len(selected) == len(reference),
        'selected_status': selected_status,
        'reference_status': reference_status,
    }
