"""Fixed sequential request protocol for the targeted scheduler recheck."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--base', default='http://127.0.0.1:19999')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=6)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('benchmark', args.benchmark)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    corpus = json.loads(args.corpus.read_text())
    assert corpus['schema'] == bench.SCHEMA and args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=False)
    initial = bench.settled_metrics(args.base)
    result = dict(schema=bench.SCHEMA, label=args.label,
        corpus_sha256=bench.digest(corpus['cases']), base=args.base,
        configuration=json.loads(args.configuration.read_text()),
        qualification=corpus['qualification'], samples=[], valid=False,
        protocol='Rotate case order by repetition; one warmup immediately before each measured request.')
    try:
        cases = corpus['cases']
        for repetition in range(args.repeats):
            offset = repetition % len(cases)
            for case in cases[offset:] + cases[:offset]:
                for warmup in (True, False):
                    before = bench.settled_metrics(args.base)
                    sample = bench.request(args.base, case['request'], None)
                    after = bench.settled_metrics(args.base)
                    delta = bench.difference(before, after)
                    sample.update(id=case['id'], repetition=repetition, warmup=warmup,
                        metrics_delta=delta)
                    result['samples'].append(sample)
                    if (delta['flare_requests_completed_total'] != 1
                        or delta['flare_requests_failed_total'] != 0
                        or delta['flare_generated_tokens_total'] != sample['usage']['completion_tokens']):
                        raise RuntimeError('Per-request metrics indicate interference; result retained as invalid')
                    if not warmup:
                        print(json.dumps(dict(case=case['id'], repetition=repetition,
                            core_tokens_s=round(sample['core_tokens_s'], 3),
                            ttft_ms=round(sample['ttft_ms'], 3))), flush=True)
                    # Publish partial measurements without reply text. Final
                    # results remain immutable; retain this journal on failure.
                    with os.fdopen(os.open(args.output / 'samples.jsonl',
                            os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), 'a') as file:
                        file.write(json.dumps(sample) + '\n')
        final = bench.settled_metrics(args.base)
        delta = bench.difference(initial, final)
        count = len(result['samples'])
        result.update(metrics_delta=delta, isolation=dict(idle_after=bench.idle(final),
            expected_requests=count, completed_requests=delta['flare_requests_completed_total'],
            failed_requests=delta['flare_requests_failed_total']))
        result['valid'] = (bench.idle(final) and delta['flare_requests_completed_total'] == count
            and delta['flare_requests_failed_total'] == 0
            and delta['flare_generated_tokens_total'] == sum(s['usage']['completion_tokens'] for s in result['samples']))
        if not result['valid']:
            raise RuntimeError('Arm isolation check failed')
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        bench.save(args.output / 'results.json', result)
    print(json.dumps(bench.summary(result)), flush=True)


if __name__ == '__main__':
    main()
