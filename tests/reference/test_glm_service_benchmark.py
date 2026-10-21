"""Benchmark integrity checks using a real local HTTP/SSE connection, no GPU."""
import argparse
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

import pytest

from qualify import glm_service_benchmark as bench


@contextmanager
def server(*, extra_completion=False, missing_timestamps=False):
    class Handler(BaseHTTPRequestHandler):
        completed = generated = active_reads = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            active = self.active_reads > 0
            type(self).active_reads = max(0, self.active_reads - 1)
            values = dict(flare_active_requests=int(active), flare_pending_requests=0,
                flare_requests_completed_total=self.completed, flare_requests_failed_total=0,
                flare_generated_tokens_total=self.generated)
            self.wfile.write("".join(f"{k} {v}\n" for k,v in values.items()).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            count = body['max_tokens']
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for offset in (0, 2):
                choice = dict(index=0, delta={'reasoning_content':'PRIVATE_OUTPUT'},
                    token_ids=list(range(offset, offset+2)), finish_reason='length' if offset else None)
                if not missing_timestamps:
                    choice['token_timestamps_ns'] = [1_000_000_000+i*50_000_000 for i in range(offset,offset+2)]
                self.wfile.write(b'data: '+json.dumps({'choices':[choice]}).encode()+b'\n\n')
                self.wfile.flush()
                time.sleep(.005)
            self.wfile.write(b'data: '+json.dumps({'choices':[], 'usage':dict(prompt_tokens=80,
                completion_tokens=count, prompt_tokens_details={'cached_tokens':64})}).encode()+b'\n\ndata: [DONE]\n\n')
            self.wfile.flush()
            type(self).completed += 2 if extra_completion else 1
            type(self).generated += count
            type(self).active_reads = 2  # retirement lags SSE DONE

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}'
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_multi_token_sse_chunks_keep_native_clock_and_omit_private_text():
    with server() as base:
        result = bench.request(base, {'max_tokens':4}, None)
    assert result['core_tokens_s'] == 20
    assert result['cached_tokens'] == 64
    assert result['usage']['completion_tokens'] == len(result['token_ids']) == 4
    assert 'PRIVATE_OUTPUT' not in json.dumps(result)


def test_missing_native_timestamps_cannot_be_reported_as_a_valid_measurement():
    with server(missing_timestamps=True) as base:
        with pytest.raises(ValueError, match='timestamps'):
            bench.request(base, {'max_tokens':4}, None)


@pytest.mark.parametrize('contaminated', [False, True])
def test_request_retirement_is_waited_and_foreign_completions_fail_isolation(tmp_path, capsys, contaminated):
    corpus = tmp_path/'corpus.json'
    bench.save(corpus, {'schema':bench.SCHEMA, 'cases':[{'id':'sample','request':{'max_tokens':4}}],
                        'qualification':'OPEN'})
    output = tmp_path/'run'
    with server(extra_completion=contaminated) as base:
        args=argparse.Namespace(corpus=corpus, repeats=1, output=output, api_key_file=None,
            metrics_base=None, base=base, label='local-protocol', configuration=None)
        if contaminated:
            with pytest.raises(RuntimeError, match='contaminated'):
                bench.run(args)
        else:
            bench.run(args)
    data=json.loads((output/'results.json').read_text())
    assert data['valid'] == (not contaminated)
    assert len(data['samples']) == 2
    assert 'PRIVATE_OUTPUT' not in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        bench.save(output/'results.json', {})
