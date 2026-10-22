"""Numeric-only comparison of targeted scheduler arms and adjacent controls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

LABELS = ['baseline-before', 'throughput3', 'baseline-middle', 'throughput4', 'baseline-after']
NEIGHBORS = {'throughput3': ['baseline-before', 'baseline-middle'],
             'throughput4': ['baseline-middle', 'baseline-after']}


def samples(arm, case=None):
    return [s for s in arm['samples'] if not s['warmup'] and (case is None or s['id'] == case)]


def key(sample, *, output=False):
    values = (sample['id'], sample['repetition'], sample['usage']['prompt_tokens'],
        sample['usage']['completion_tokens'], sample['cached_tokens'], tuple(sample['finish']))
    return values + (sample['output_sha256'],) if output else values


def output_key(sample):
    return (sample['id'], sample['usage']['prompt_tokens'], sample['usage']['completion_tokens'],
        sample['cached_tokens'], tuple(sample['finish']), sample['output_sha256'])


def summarize(group):
    counters = Counter()
    for sample in group:
        counters.update(sample['metrics_delta'])
    rounds, gpu_ns, depths = {}, {}, defaultdict(dict)
    for name, value in counters.items():
        family = name.split('{')[0]
        labels = dict(re.findall(r'(\w+)="([^"]+)"', name))
        if family in ['flare_speculative_rounds_total', 'flare_speculative_gpu_nanoseconds_total'] and value:
            shape = 'b' + labels['batch'] + 'w' + labels['width']
            (rounds if family.endswith('_rounds_total') else gpu_ns)[shape] = value
        if family == 'flare_speculative_drafts_total' and labels.get('source') == 'mtp':
            depths[labels['depth']][labels['outcome']] = value
    tps = [s['core_tokens_s'] for s in group]
    return dict(measured_requests=len(group), tps_median=statistics.median(tps),
        tps_min=min(tps), tps_max=max(tps),
        tps_samples=tps,
        zero_accepted_draft_requests=sum(s['metrics_delta'].get('flare_accepted_draft_tokens_total', 0) == 0 for s in group),
        end_to_end_tps_median=statistics.median(s['end_to_end_tokens_s'] for s in group),
        ttft_ms_median=statistics.median(s['ttft_ms'] for s in group),
        distinct_outputs=len({s['output_sha256'] for s in group}), rounds=rounds,
        average_round_gpu_ms={shape:gpu_ns[shape] / count / 1e6 for shape,count in rounds.items() if shape in gpu_ns},
        accepted_drafts=counters['flare_accepted_draft_tokens_total'],
        draft_request_rounds=counters['flare_draft_request_rounds_total'],
        depth_outcomes=dict(depths))


def report(root):
    arms = {name:json.loads((root / name / 'results.json').read_text()) for name in LABELS}
    assert all(arm['valid'] for arm in arms.values())
    assert len({arm['corpus_sha256'] for arm in arms.values()}) == 1
    cases = list(dict.fromkeys(s['id'] for s in samples(arms[LABELS[0]])))
    summaries = {name:{case:summarize(samples(arm,case)) for case in cases} for name,arm in arms.items()}
    comparisons = {}
    for label,neighbors in NEIGHBORS.items():
        candidate = arms[label]
        outputs = {}
        for case in cases:
            current = samples(candidate, case)
            reference = [s for name in neighbors for s in samples(arms[name], case)]
            by_output = defaultdict(list)
            for sample in reference:
                by_output[output_key(sample)].append(sample['core_tokens_s'])
            matching = [s for s in current if output_key(s) in by_output]
            ratios = [s['core_tokens_s'] / statistics.median(by_output[output_key(s)]) for s in matching]
            paired = {}
            for name in neighbors:
                old = samples(arms[name],case)
                assert [(s['id'],s['repetition']) for s in old] == [(s['id'],s['repetition']) for s in current]
                paired[name] = dict(work_cache_match=all(key(a)==key(b) for a,b in zip(old,current)),
                    exact_output_pairs=sum(a['token_ids']==b['token_ids'] for a,b in zip(old,current)), pairs=len(current))
            outputs[case] = dict(candidate_tps=summaries[label][case]['tps_median'],
                neighbor_controls_tps=[summaries[name][case]['tps_median'] for name in neighbors],
                neighbor_output_variants=len(by_output), candidate_output_variants=len({s['output_sha256'] for s in current}),
                exact_output_match_to_any_neighbor=len(matching), candidate_samples=len(current),
                exact_output_subset_tps_ratio_median=statistics.median(ratios) if ratios else None,
                paired=paired)
        comparisons[label] = outputs
    return dict(root=str(root), arm_summaries=summaries, comparisons=comparisons,
        requests_including_warmups=sum(len(a['samples']) for a in arms.values()),
        qualification='Diagnostic only; ordinary/DFlash2 qualification remains OPEN. Exact-output subsets cannot clear the full-model gate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = report(args.root)
    with args.output.open('x') as file:
        json.dump(data,file,indent=2)
    print(json.dumps(dict(arms=len(data['arm_summaries']), requests=data['requests_including_warmups'], output=str(args.output))))
