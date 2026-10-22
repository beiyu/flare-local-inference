"""Real HTTP/SSE isolation and work-matched numeric-summary checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from qualify import glm_scheduler_recheck as runner
from qualify import glm_scheduler_recheck_summary as summary
from qualify import glm_service_benchmark as bench
from tests.reference.test_glm_service_benchmark import server


def run_protocol(tmp_path, monkeypatch, contaminated=False):
    bench.save(tmp_path/'corpus.json', dict(schema=bench.SCHEMA,
        cases=[dict(id='case', request=dict(max_tokens=4))], qualification='OPEN'))
    bench.save(tmp_path/'config.json', {})
    with server(extra_completion=contaminated) as base:
        monkeypatch.setattr(sys, 'argv', ['runner', '--benchmark', str(Path(bench.__file__)),
            '--corpus', str(tmp_path/'corpus.json'), '--configuration', str(tmp_path/'config.json'),
            '--output', str(tmp_path/'original'), '--base', base, '--label', 'protocol', '--repeats', '2'])
        if contaminated:
            with pytest.raises(RuntimeError, match='interference'):
                runner.main()
        else:
            runner.main()
    return json.loads((tmp_path/'original/results.json').read_text())


@pytest.mark.parametrize('contaminated', [False, True])
def test_per_request_counter_isolation_and_private_partial_journal(tmp_path, monkeypatch, capsys, contaminated):
    result = run_protocol(tmp_path, monkeypatch, contaminated)
    assert result['valid'] == (not contaminated)
    assert (tmp_path/'original/results.json').stat().st_mode & 0o777 == 0o600
    if not contaminated:
        assert len(result['samples']) == 4
        assert all(s['metrics_delta']['flare_requests_completed_total'] == 1 for s in result['samples'])
        assert (tmp_path/'original/samples.jsonl').stat().st_mode & 0o777 == 0o600
    assert 'PRIVATE_OUTPUT' not in capsys.readouterr().out


@pytest.mark.parametrize('cache_mismatch', [False, True])
def test_equal_output_subset_requires_equal_cache_and_work(tmp_path, monkeypatch, cache_mismatch):
    original = run_protocol(tmp_path, monkeypatch)
    for label in summary.LABELS:
        data = deepcopy(original)
        data['label'] = label
        if label == 'throughput3':
            measured = summary.samples(data)
            measured[0]['core_tokens_s'] = 30
            measured[1]['core_tokens_s'] = 40
            measured[1]['token_ids'] = [9, 8, 7, 6]
            measured[1]['output_sha256'] = bench.digest(measured[1]['token_ids'])
            if cache_mismatch:
                measured[0]['cached_tokens'] = 0
        directory = tmp_path/label
        directory.mkdir()
        bench.save(directory/'results.json', data)
    result = summary.report(tmp_path)['comparisons']['throughput3']['case']
    assert result['candidate_tps'] == 35
    assert result['exact_output_match_to_any_neighbor'] == (0 if cache_mismatch else 1)
    assert result['exact_output_subset_tps_ratio_median'] == (None if cache_mismatch else 1.5)
    assert all(p['work_cache_match'] == (not cache_mismatch) for p in result['paired'].values())
