"""Helper to save batch judgments made by Claude."""
import json, sys, os

PROG_PATH = r'C:\dev\TalkTrip\TripTalk_ai_model\dataset\test-add-aihub-progress.json'
IN_PATH = r'C:\dev\TalkTrip\TripTalk_ai_model\dataset\test-add-aihub.json'

def save_batch(source_name, start_idx, judgments):
    """
    judgments: list of (label, confidence) for each msg in source order
    start_idx: global starting index for this source
    """
    data = json.load(open(IN_PATH, encoding='utf-8'))
    progress = json.load(open(PROG_PATH, encoding='utf-8'))

    # Get indices for this source
    indices = [i for i, d in enumerate(data) if d['source'] == source_name]
    assert len(indices) == len(judgments), f"Mismatch: source has {len(indices)} msgs, got {len(judgments)} judgments"

    for idx, (label, conf) in zip(indices, judgments):
        progress['results'].append({
            'id': f'aihub_{idx+1:05d}',
            'source': source_name,
            'text': data[idx]['text'],
            'label': label,
            'confidence': conf,
        })

    progress['processed_count'] = len(progress['results'])
    if source_name not in progress['sources_processed']:
        progress['sources_processed'].append(source_name)

    with open(PROG_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

    from collections import Counter
    batch_labels = [j[0] for j in judgments]
    batch_conf = [j[1] for j in judgments]
    return {
        'processed': progress['processed_count'],
        'sources_done': len(progress['sources_processed']),
        'batch_dist': dict(Counter(batch_labels)),
        'batch_conf': dict(Counter(batch_conf)),
    }

if __name__ == '__main__':
    import ast
    src = sys.argv[1]
    judgments = ast.literal_eval(sys.argv[2])
    stats = save_batch(src, 0, judgments)
    print(stats)
