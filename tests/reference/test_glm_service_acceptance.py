"""Offline client/validator tests, not GPU or live service acceptance results."""
import base64
import copy
import io
import json
import struct
import zlib

import pytest

from qualify import glm_service_acceptance as api


def png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress(b"\0\xff\0\0")) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def response():
    return {"model": api.ALIAS, "choices": [{"index": 0, "message": {"role": "assistant", "content": "PRIVATE_ANSWER"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8}}


@pytest.fixture
def baselines(tmp_path):
    for image, name in ((False, "text"), (True, "image")):
        request = {"model": api.ALIAS, "temperature": 0, "max_tokens": 128, "stream": False,
            "messages": [{"role": "user", "content": "PRIVATE_PROMPT"}], "chat_template_kwargs": {"enable_thinking": False}}
        result = response()
        if image:
            request["messages"][0]["content"] = [{"type": "text", "text": "PRIVATE_IMAGE_PROMPT"}, {"type": "image_url", "image_url": {"url": png()}}]
            result["usage"]["prompt_tokens_details"] = {"multimodal_tokens": {"image": 1}}
        (tmp_path / (name + ".json")).write_text(json.dumps({"request": request, "response": result, "seconds": .1}))
    return tmp_path


def test_plan_reads_real_envelopes_without_http_or_payload_disclosure(baselines, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("plan must not instantiate an HTTP client")
    monkeypatch.setattr(api, "Endpoint", forbidden)
    output = baselines / "plan-result.json"
    assert api.main(["plan", "--baseline", str(baselines), "--output", str(output)]) == 0
    raw = capsys.readouterr().out
    report = json.loads(raw)
    assert report["executed_http"] is False and report["mode"] == "plan"
    assert report["baseline"]["image"]["images"] == 1
    assert "PRIVATE" not in raw and "base64" not in raw and png() not in raw
    assert "core_passed" not in raw and "cases" not in raw
    assert json.loads(output.read_text()) == report
    assert api.main(["plan", "--baseline", str(baselines), "--output", str(output)]) == 2


@pytest.mark.parametrize("mutation", ["external_image", "bad_png", "auth", "model", "temperature", "accounting"])
def test_baseline_safety_and_reference_checks(baselines, mutation):
    path = baselines / "image.json"; value = json.loads(path.read_text())
    if mutation == "external_image": value["request"]["messages"][0]["content"][1]["image_url"]["url"] = "https://example.invalid/private.png"
    elif mutation == "bad_png": value["request"]["messages"][0]["content"][1]["image_url"]["url"] = "data:image/png;base64,AAAA"
    elif mutation == "auth": value["request"]["Authorization"] = "PRIVATE_CREDENTIAL"
    elif mutation == "model": value["request"]["model"] = "wrong"
    elif mutation == "temperature": value["request"]["temperature"] = 1
    else: value["response"]["usage"]["total_tokens"] = 0
    path.write_text(json.dumps(value))
    with pytest.raises(api.CheckFailure): api.load_baseline(path, image=True, model=api.ALIAS)


def wire(events):
    return io.BytesIO(b"".join(("data: " + (event if isinstance(event, str) else json.dumps(event)) + "\r\n\r\n").encode() for event in events))


def test_sse_utf8_and_fragmented_tool_arguments_reconstruct_real_chat_shape():
    events = [
        {"model": api.ALIAS, "choices": [{"index": 0, "delta": {"reasoning_content": "思考", "content": "PRIVATE_"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "ANSWER"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": response()["usage"]}, "[DONE]",
    ]
    result = api.aggregate_stream(api.sse_events(wire(events)))
    api.validate_response(result, api.ALIAS)
    assert api.answer(result) == "PRIVATE_ANSWER" and api.reasoning(result) == "思考"
    calls = [
        {"model": api.ALIAS, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "migration_", "arguments": '{"value":'}}]}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"name": "echo", "arguments": '"FLARE_TOOL_OK"}'}}]}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": response()["usage"]}, "[DONE]",
    ]
    call_response = api.aggregate_stream(api.sse_events(wire(calls)))
    api.validate_response(call_response, api.ALIAS); api.validate_tools(call_response, "auto")
    with pytest.raises(api.CheckFailure): api.validate_tools(call_response, "none")


@pytest.mark.parametrize("events", [
    [], ["[DONE]"], ["[DONE]", "[DONE]"],
    [{"error": {"message": "PRIVATE_ERROR"}}],
    [{"model": api.ALIAS, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}],
    ["[DONE]", {"choices": []}],
])
def test_incomplete_error_or_extra_terminal_sse_never_passes(events):
    with pytest.raises(api.CheckFailure): api.aggregate_stream(api.sse_events(wire(events)))


def test_reference_comparison_is_strict_and_preserves_requests(baselines):
    baseline = api.load_baseline(baselines / "text.json", image=False, model=api.ALIAS)
    before = copy.deepcopy(baseline.request)
    api.compare_baseline(copy.deepcopy(baseline.response), baseline)
    drift = copy.deepcopy(baseline.response); drift["choices"][0]["message"]["content"] += " "
    with pytest.raises(api.CheckFailure): api.compare_baseline(drift, baseline)
    for choice in ("none", "auto"):
        built = api.tools_request(baseline, choice, stream=True)
        assert built["temperature"] == 0 and built["tool_choice"] == choice and built["stream_options"]["include_usage"]
    assert baseline.request == before


@pytest.mark.parametrize("url", ["file:///private/request", "http://user:PRIVATE@example.invalid", "http://localhost/?key=PRIVATE", "http://localhost/#PRIVATE"])
def test_endpoint_never_accepts_credentials_or_secret_query_fields(url):
    with pytest.raises(api.CheckFailure): api.Endpoint(url)


def test_empty_key_matches_upstream_and_embedded_newlines_are_rejected(tmp_path):
    key = tmp_path / "key"; key.write_text("\n")
    assert api.Endpoint("http://127.0.0.1:9999", key).key == ""
    key.write_text("PRIVATE\nHEADER")
    with pytest.raises(api.CheckFailure): api.Endpoint("http://127.0.0.1:9999", key)


def test_plan_rejects_secret_endpoint_before_emitting_it(baselines, capsys):
    assert api.main(["plan", "--baseline", str(baselines), "--native-url", "http://user:PRIVATE@localhost"]) == 2
    output = capsys.readouterr().out
    assert "PRIVATE" not in output and "ready_for_live_run" not in output


def test_measurement_summary_does_not_copy_arbitrary_upstream_fields():
    result = response()
    result["usage"]["private"] = "PRIVATE_CREDENTIAL"
    result["usage"]["prompt_tokens_details"] = {"private": "PRIVATE_PROMPT", "cached_tokens": 2,
                                               "multimodal_tokens": {"image": 16, "private": "PRIVATE"}}
    recorded = api.summary(result)
    assert "PRIVATE" not in json.dumps(recorded)
    assert recorded["usage"]["prompt_tokens_details"] == {"cached_tokens": 2, "multimodal_tokens": {"image": 16}}


def test_client_performs_real_local_http_and_sse_without_double_v1_or_metrics_prefix():
    import http.server
    import threading
    received = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            received.append(self.path)
            if self.path != "/metrics?format=prometheus":
                self.send_response(200); self.end_headers()
                self.wfile.write(b'{"cancelled": 2}')
                return
            self.send_response(200); self.end_headers()
            self.wfile.write(b"flare_requests_cancelled_total 2\n")
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.end_headers()
            if body.get("stream"):
                events = [{"model": api.ALIAS, "choices": [{"index": 0, "delta": {"content": "PRIVATE_ANSWER"}, "finish_reason": "stop"}]},
                          {"choices": [], "usage": response()["usage"]}, "[DONE]"]
                self.wfile.write(wire(events).read())
            else:
                self.wfile.write(json.dumps(response()).encode())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        endpoint = api.Endpoint(f"http://127.0.0.1:{server.server_port}/v1")
        assert api.metric(endpoint, "flare_requests_cancelled_total") == 2
        for stream in (False, True):
            result = endpoint.chat({"messages": [], "stream": stream})
            api.validate_response(result, api.ALIAS)
            assert api.answer(result) == "PRIVATE_ANSWER"
        assert received[0] == "/metrics?format=prometheus"
        assert [item[0] for item in received[1:]] == ["/v1/chat/completions"] * 2
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


@pytest.mark.parametrize("count,status,count_error,expected", [
    (20, 200, False, True), (21, 400, True, True),
    (21, 400, False, False), (21, 200, False, False), (20, 400, False, False),
])
def test_heavy_image_observations_require_real_completion_or_specific_count_error(
        baselines, count, status, count_error, expected):
    from types import SimpleNamespace
    baseline = api.load_baseline(baselines / "image.json", image=True, model=api.ALIAS)
    captured = []
    message = "request has 21 images; max_images_per_prompt is 20" if count_error else "HTTP body limit exceeded"
    result = response() if status == 200 else {"error": {"message": message}}
    class Reply:
        def __init__(self): self.status = status
        def read(self, _): return json.dumps(result).encode()
        def close(self): pass
    def opened(path, body):
        captured.append(body)
        return SimpleNamespace(close=lambda: None), Reply()
    endpoint = SimpleNamespace(open=opened)
    suite = api.Suite(endpoint, endpoint, {"image": baseline}, SimpleNamespace(model=api.ALIAS))
    observed = suite.heavy_images(count)
    assert observed["mode"] == "observation_only" and observed["observed_http_status"] == status
    assert observed["reference_limit_match"] is expected
    assert captured[0]["max_tokens"] == 8
    assert sum(part["type"] == "image_url" for part in captured[0]["messages"][0]["content"]) == count
    assert "PRIVATE" not in json.dumps(observed)


def test_gateway_model_is_explicit_in_plan_without_rewriting_native_baseline(baselines, capsys):
    gateway = "spark/" + api.ALIAS
    assert api.main(["plan", "--baseline", str(baselines), "--gateway-model", gateway]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == api.ALIAS and report["gateway_model"] == gateway
    assert report["gateway_response_model"] == api.ALIAS
    assert report["endpoints"][1] == "http://127.0.0.1:8889"
    assert report["executed_http"] is False
    assert json.loads((baselines / "text.json").read_text())["request"]["model"] == api.ALIAS
    assert api.main(["plan", "--baseline", str(baselines), "--gateway-model", ""]) == 2
    capsys.readouterr()
    assert api.main(["plan", "--baseline", str(baselines), "--gateway-model", gateway,
                     "--gateway-response-model", "explicit/upstream"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["gateway_model"] == gateway and report["gateway_response_model"] == "explicit/upstream"
    assert api.main(["plan", "--baseline", str(baselines), "--gateway-response-model", ""]) == 2


def test_native_and_gateway_aliases_use_real_http_and_preserve_reference_semantics(baselines, capsys):
    # Two local protocol fixtures exercise the client/Suite only. They are not
    # model, gateway deployment or GPU acceptance evidence.
    import http.server
    import threading
    from types import SimpleNamespace
    references = {name: api.load_baseline(baselines / (name + ".json"), image=name == "image", model=api.ALIAS)
                  for name in ("text", "image")}
    originals = copy.deepcopy(references)
    gateway_model = "spark/" + api.ALIAS
    received = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            if self.path == "/metrics?format=prometheus":
                self.send_response(200); self.end_headers()
                self.wfile.write(b"flare_requests_cancelled_total 0\n")
                return
            assert self.path == "/v1/models"
            card = {"id": self.server.alias, "flare": {"default_max_tokens": 65536,
                     "max_images_per_prompt": 20, "max_concurrency": 4}}
            self.send_response(200); self.end_headers()
            self.wfile.write(json.dumps({"object": "list", "data": [card]}).encode())
        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.server.alias, body))
            count = sum(part.get("type") == "image_url" for message in body["messages"]
                        if isinstance(message.get("content"), list) for part in message["content"])
            if count == 21:
                self.send_response(400); self.end_headers()
                self.wfile.write(json.dumps({"error": {"message": "request has 21 images; max_images_per_prompt is 20"}}).encode())
                return
            result = copy.deepcopy(references["image" if count else "text"].response)
            result["model"] = self.server.response_alias
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
            self.end_headers()
            if body.get("stream"):
                events = [{"model": result["model"], "choices": [{"index": 0,
                    "delta": result["choices"][0]["message"], "finish_reason": result["choices"][0]["finish_reason"]}]},
                    {"model": result["model"], "choices": [], "usage": result["usage"]}, "[DONE]"]
                self.wfile.write(wire(events).read())
            else:
                self.wfile.write(json.dumps(result).encode())
    servers, threads, endpoints = [], [], []
    try:
        for alias in (api.ALIAS, gateway_model):
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.alias, server.response_alias = alias, api.ALIAS
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            servers.append(server); threads.append(thread)
            endpoints.append(api.Endpoint(f"http://127.0.0.1:{server.server_port}"))
        args = SimpleNamespace(model=api.ALIAS)
        native = api.Suite(endpoints[0], endpoints[0], references, args)
        gateway = api.Suite(endpoints[1], endpoints[0], references, args, model=gateway_model, expected_response_model=api.ALIAS)
        for suite in (native, gateway):
            assert suite.models()["alias_present"] is True
            for baseline in references.values():
                for stream in (False, True):
                    result = suite.baseline_case(baseline, stream=stream)
                    assert result["model"] == suite.expected_response_model == api.ALIAS
            for count in (20, 21):
                assert suite.heavy_images(count)["reference_limit_match"] is True
        # These static reply fixtures cannot establish tools/reasoning/cancellation,
        # so a full run must still fail while its real HTTP baseline cases pass.
        assert api.main(["run", "--baseline", str(baselines), "--native-url", endpoints[0].base,
                         "--gateway-url", endpoints[1].base, "--gateway-model", gateway_model,
                         "--cancel-wait", "0.001"]) == 1
        raw = capsys.readouterr().out
        recorded = json.loads(raw)
        for key, request_alias in (("native", api.ALIAS), ("gateway", gateway_model)):
            report = recorded[key]
            assert report["model"] == request_alias and report["expected_response_model"] == api.ALIAS
            assert report["observed_response_models"] == [api.ALIAS]
            assert report["core_passed"] is False
            assert all(case["status"] == "passed" for case in report["cases"]
                       if case["case"] in ("text_temperature_zero", "image_temperature_zero", "text_streaming", "image_streaming"))
        assert "PRIVATE" not in raw
        assert all(body["model"] == alias for alias, body in received)
        assert {alias for alias, body in received} == {api.ALIAS, gateway_model}
        assert references == originals
        assert native.model == native.expected_response_model == api.ALIAS
        assert gateway.model == gateway_model and gateway.expected_response_model == api.ALIAS
        assert native.observed_models == gateway.observed_models == {api.ALIAS}
        for suite, server in zip((native, gateway), servers):
            for wrong_alias in (gateway_model, "unrelated-model"):
                server.response_alias = wrong_alias
                for stream in (False, True):
                    with pytest.raises(api.CheckFailure, match="response model"):
                        suite.baseline_case(references["text"], stream=stream)
                assert suite.heavy_images(20)["reference_limit_match"] is False
            server.response_alias = api.ALIAS
            assert suite.observed_models == {api.ALIAS, gateway_model, "unrelated-model"}
        servers[1].response_alias = "explicit/upstream"
        explicit = api.Suite(endpoints[1], endpoints[0], references, args, model=gateway_model,
                             expected_response_model="explicit/upstream")
        for stream in (False, True):
            assert explicit.baseline_case(references["text"], stream=stream)["model"] == "explicit/upstream"
        assert explicit.heavy_images(20)["reference_limit_match"] is True
        assert api.main(["run", "--baseline", str(baselines), "--native-url", endpoints[0].base,
                         "--gateway-url", endpoints[1].base, "--gateway-model", gateway_model,
                         "--gateway-response-model", "explicit/upstream", "--cancel-wait", "0.001"]) == 1
        recorded = json.loads(capsys.readouterr().out)
        assert recorded["gateway"]["model"] == gateway_model
        assert recorded["gateway"]["expected_response_model"] == "explicit/upstream"
        assert recorded["gateway"]["observed_response_models"] == ["explicit/upstream"]
        assert all(case["status"] == "passed" for case in recorded["gateway"]["cases"]
                   if case["case"] in ("text_temperature_zero", "image_temperature_zero", "text_streaming", "image_streaming"))
        servers[1].alias = api.ALIAS
        with pytest.raises(api.CheckFailure, match="absent"):
            gateway.models()
    finally:
        for server in servers: server.shutdown(); server.server_close()
        for thread in threads: thread.join(timeout=2)
