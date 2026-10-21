"""Check the deployed GLM Chat API using existing text/image baseline files.

No model or CUDA imports. plan validates inputs without HTTP. run performs real
requests and emits only measurements/check outcomes, never prompts, image bytes,
answer text or authentication values. It does not change accounts or services.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import socket
import struct
import threading
import time
from urllib.parse import urlsplit

LIMIT = 4 << 20
ALIAS = "GLM-5.3-Flash-EXL3"


class CheckFailure(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


@dataclass
class Baseline:
    request: dict
    response: dict
    images: int


def png_data(url):
    require(isinstance(url, str) and url.startswith("data:image/png;base64,"), "baseline image must be an embedded PNG")
    try:
        raw = base64.b64decode(url.split(",", 1)[1], validate=True)
    except (ValueError, TypeError) as error:
        raise CheckFailure("baseline PNG has invalid base64") from error
    require(24 <= len(raw) <= LIMIT and raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR", "baseline PNG signature/header is invalid")
    width, height = struct.unpack(">II", raw[16:24])
    require(width > 0 and height > 0, "baseline PNG dimensions are invalid")
    return {"bytes": len(raw), "width": width, "height": height}


def load_baseline(path, *, image, model):
    require(path.is_file() and path.stat().st_size <= LIMIT, "baseline file is missing or too large")
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise CheckFailure("baseline JSON is invalid") from error
    require(isinstance(value, dict), "baseline must be a JSON object")
    request, response = value.get("request"), value.get("response")
    require(isinstance(request, dict) and isinstance(response, dict), "baseline needs request and response objects")
    require(not {str(k).lower() for k in request} & {"api_key", "authorization", "headers", "password", "access_token"}, "baseline must not contain authentication fields")
    require(request.get("model") == model and request.get("temperature") == 0, "baseline must use the requested alias and temperature zero")
    require(request.get("stream") is False and isinstance(request.get("max_tokens"), int) and 0 < request["max_tokens"] <= 4096, "baseline must be buffered with a bounded output limit")
    messages = request.get("messages")
    require(isinstance(messages, list) and messages, "baseline needs messages")
    images = 0
    for message in messages:
        require(isinstance(message, dict) and isinstance(message.get("role"), str), "baseline message is invalid")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                require(isinstance(part, dict), "baseline content part is invalid")
                if part.get("type") == "image_url":
                    png_data(part.get("image_url", {}).get("url")); images += 1
                else:
                    require(part.get("type") == "text" and isinstance(part.get("text"), str), "baseline content part is unsupported")
        else:
            require(isinstance(content, str), "baseline message content is invalid")
    require(images == int(image), "text baseline must have no images and image baseline exactly one")
    validate_response(response, model)
    require(bool(answer(response)), "baseline answer must be nonempty")
    return Baseline(copy.deepcopy(request), response, images)


def answer(response):
    return response["choices"][0].get("message", {}).get("content") or ""


def reasoning(response):
    return response["choices"][0].get("message", {}).get("reasoning_content") or ""


def validate_response(response, model):
    require(isinstance(response, dict) and "error" not in response, "API returned an error object")
    require(response.get("model") == model, "response model differs from the expected deployed alias")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1 and choices[0].get("index") == 0, "API must return exactly choice zero")
    require(choices[0].get("finish_reason") in ("stop", "length", "tool_calls"), "completion has no valid terminal reason")
    message = choices[0].get("message")
    require(isinstance(message, dict) and isinstance(message.get("content") or "", str), "completion message is invalid")
    usage = response.get("usage")
    require(isinstance(usage, dict), "completion usage is missing")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(field)) is int and usage[field] >= 0, "completion usage must be nonnegative integers")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "completion token accounting is inconsistent")
    return response


def compare_baseline(response, baseline):
    require(answer(response) == answer(baseline.response), "answer differs from the recorded baseline")
    require(reasoning(response) == reasoning(baseline.response), "reasoning differs from the recorded baseline")
    require(response["usage"]["prompt_tokens"] == baseline.response["usage"]["prompt_tokens"], "rendered prompt token count differs from baseline")
    require(response["usage"]["completion_tokens"] == baseline.response["usage"]["completion_tokens"], "generated token count differs from baseline")
    require(response["choices"][0]["finish_reason"] == baseline.response["choices"][0]["finish_reason"], "termination differs from baseline")
    if baseline.images:
        def image_tokens(usage):
            details = usage.get("prompt_tokens_details")
            multimodal = details.get("multimodal_tokens") if isinstance(details, dict) else None
            return multimodal.get("image") if isinstance(multimodal, dict) else None
        expected, actual = image_tokens(baseline.response["usage"]), image_tokens(response["usage"])
        require(type(expected) is int and type(actual) is int and actual == expected, "image token accounting differs from baseline")


def sse_events(source):
    data, consumed = [], 0
    while True:
        line = source.readline(LIMIT + 1)
        if not line:
            require(not data, "SSE ended before its event separator")
            return
        consumed += len(line)
        require(consumed <= LIMIT, "SSE exceeded the response byte limit")
        try:
            text = line.decode("utf-8").rstrip("\r\n")
        except UnicodeError as error:
            raise CheckFailure("SSE is not UTF-8") from error
        if not text:
            if not data:
                continue
            payload = "\n".join(data); data.clear()
            if payload == "[DONE]":
                yield "[DONE]"
            else:
                try:
                    event = json.loads(payload)
                except ValueError as error:
                    raise CheckFailure("SSE data is not JSON") from error
                require(isinstance(event, dict) and "error" not in event, "SSE contains a terminal error")
                yield event
        elif text.startswith("data:"):
            data.append(text[5:].lstrip(" "))


def aggregate_stream(events):
    message, calls, finishes, usage, model, done = {"role": "assistant", "content": ""}, {}, [], None, None, 0
    for event in events:
        if event == "[DONE]":
            done += 1
            continue
        require(done == 0, "SSE emitted data after DONE")
        if event.get("model") is not None:
            require(model in (None, event["model"]), "SSE changed model identity")
            model = event["model"]
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            require(choice.get("index") == 0, "SSE returned an unexpected choice index")
            delta = choice.get("delta", {})
            for field in ("content", "reasoning_content"):
                if delta.get(field) is not None:
                    require(isinstance(delta[field], str), "SSE text delta is invalid")
                    message[field] = message.get(field, "") + delta[field]
            for call in delta.get("tool_calls", []):
                index = call.get("index")
                require(type(index) is int and index >= 0, "SSE tool index is invalid")
                target = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if call.get("id"):
                    target["id"] = call["id"]
                for field in ("name", "arguments"):
                    value = call.get("function", {}).get(field, "")
                    require(isinstance(value, str), "SSE tool delta is invalid")
                    target["function"][field] += value
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    require(done == 1 and len(finishes) == 1, "SSE needs exactly one terminal reason and DONE")
    if calls:
        require(sorted(calls) == list(range(len(calls))), "SSE tool indices contain a gap")
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": finishes[0]}], "usage": usage}


def endpoint_url(url):
    parsed = urlsplit(url)
    require(parsed.scheme in ("http", "https") and parsed.hostname is not None
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment
            and parsed.path.rstrip("/") in ("", "/v1"),
            "endpoint must be a root or /v1 HTTP(S) URL without credentials/query")
    require(parsed.port is None or 0 < parsed.port < 65536, "endpoint port is invalid")
    return parsed


class Endpoint:
    def __init__(self, url, key_file=None, *, timeout=180):
        self.url = endpoint_url(url)
        self.base = url.rstrip("/")
        self.timeout = timeout
        self.key = Path(key_file).read_text().strip() if key_file else ""
        require("\r" not in self.key and "\n" not in self.key, "API key file has invalid line breaks")

    def open(self, path, body=None):
        connection = (http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection)(self.url.hostname, self.url.port, timeout=self.timeout)
        prefix = self.url.path.rstrip("/")
        if prefix == "/v1":
            prefix = ""
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream" if body and body.get("stream") else "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        try:
            connection.request("POST" if body is not None else "GET", prefix + path,
                body=None if body is None else json.dumps(body, ensure_ascii=False).encode(), headers=headers)
            return connection, connection.getresponse()
        except BaseException:
            connection.close()
            raise

    def get(self, path):
        connection, response = self.open(path)
        try:
            raw = response.read(LIMIT + 1)
            require(len(raw) <= LIMIT, "response exceeded the byte limit")
            return response.status, raw
        finally:
            response.close(); connection.close()

    def chat(self, body):
        connection, response = self.open("/v1/chat/completions", body)
        try:
            if response.status != 200:
                response.read(LIMIT + 1)
                raise CheckFailure(f"chat returned HTTP {response.status}")
            if body.get("stream"):
                require(response.getheader("Content-Type", "").startswith("text/event-stream"), "stream response is not SSE")
                return aggregate_stream(sse_events(response))
            raw = response.read(LIMIT + 1)
            require(len(raw) <= LIMIT, "response exceeded the byte limit")
            return json.loads(raw)
        finally:
            response.close(); connection.close()


def summary(response):
    # Keep only numeric token measurements, even if an upstream adds arbitrary
    # response fields. Never copy a response payload into the saved result.
    source = response["usage"]
    usage = {field: source[field] for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
    details = source.get("prompt_tokens_details")
    if isinstance(details, dict):
        usage["prompt_tokens_details"] = {key: value for key, value in details.items()
                                          if key == "cached_tokens" and type(value) is int}
        images = details.get("multimodal_tokens")
        if isinstance(images, dict) and type(images.get("image")) is int:
            usage["prompt_tokens_details"]["multimodal_tokens"] = {"image": images["image"]}
    details = source.get("completion_tokens_details")
    if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
        usage["completion_tokens_details"] = {"reasoning_tokens": details["reasoning_tokens"]}
    return {"model": response["model"], "finish_reason": response["choices"][0]["finish_reason"],
        "answer_chars": len(answer(response)), "reasoning_chars": len(reasoning(response)),
        "tool_calls": len(response["choices"][0]["message"].get("tool_calls", [])), "usage": usage}


def tools_request(baseline, choice, *, stream=False):
    body = copy.deepcopy(baseline.request)
    body.update(messages=[{"role": "user", "content": 'Call migration_echo with value "FLARE_TOOL_OK". Use a tool call and do not answer directly.'}],
        tools=[{"type": "function", "function": {"name": "migration_echo", "description": "Echo the supplied test value.",
            "parameters": {"type": "object", "properties": {"value": {"type": "string", "enum": ["FLARE_TOOL_OK"]}}, "required": ["value"], "additionalProperties": False}}}],
        tool_choice=choice, stream=stream)
    body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": False}
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def validate_tools(response, choice):
    calls = response["choices"][0]["message"].get("tool_calls", [])
    if choice == "none":
        require(not calls and bool(answer(response).strip()), "tool_choice none emitted tools or an empty answer")
        require(response["choices"][0]["finish_reason"] != "tool_calls", "tool_choice none has a tool terminal reason")
    else:
        require(len(calls) == 1 and calls[0].get("id"), "auto request did not produce its requested tool call")
        function = calls[0].get("function", {})
        require(function.get("name") == "migration_echo", "auto request selected the wrong tool")
        try:
            arguments = json.loads(function.get("arguments", ""))
        except (ValueError, TypeError) as error:
            raise CheckFailure("auto tool arguments are not JSON") from error
        require(arguments == {"value": "FLARE_TOOL_OK"}, "auto tool arguments differ")
        require(response["choices"][0]["finish_reason"] == "tool_calls", "auto tool call has no tool terminal reason")


def metric(endpoint, name):
    status, raw = endpoint.get("/metrics?format=prometheus")
    require(status == 200, "native metrics endpoint is unavailable")
    for line in raw.decode("utf-8").splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise CheckFailure("required native cancellation metric is missing")


def cancelled_request(endpoint, metrics, body, *, wait):
    before = metric(metrics, "flare_requests_cancelled_total")
    body = copy.deepcopy(body)
    body.update(messages=[{"role": "user", "content": "Count from one to one thousand, one number per line, without skipping numbers."}], max_tokens=512, stream=True)
    body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": False}
    connection, response = endpoint.open("/v1/chat/completions", body)
    observed = False
    try:
        require(response.status == 200, "cancellation stream was not admitted")
        for event in sse_events(response):
            require(event != "[DONE]", "cancellation stream finished before a cancellable text delta")
            if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content") for c in event.get("choices", [])):
                observed = True
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                break
    finally:
        response.close(); connection.close()
    require(observed, "cancellation did not observe a text delta")
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        delta = metric(metrics, "flare_requests_cancelled_total") - before
        if delta >= 1:
            return {"client_disconnected_after_delta": True, "native_cancellations_delta": delta}
        time.sleep(.2)
    raise CheckFailure("native cancellation counter did not advance after disconnect")


class Suite:
    def __init__(self, endpoint, metrics, baselines, args, *, model=None, expected_response_model=None):
        self.endpoint, self.metrics, self.baselines, self.args = endpoint, metrics, baselines, args
        self.model = args.model if model is None else model
        self.expected_response_model = self.model if expected_response_model is None else expected_response_model
        self.results = []
        self.observed_models = set()

    def case(self, name, function):
        started = time.monotonic()
        try:
            details = function()
            self.results.append({"case": name, "status": "passed", "seconds": round(time.monotonic() - started, 3), "details": details})
            return details
        except Exception as error:
            # Only our static check descriptions are printable; transport/JSON
            # exceptions may contain response snippets and must stay redacted.
            self.results.append({"case": name, "status": "failed", "seconds": round(time.monotonic() - started, 3),
                "reason": str(error) if isinstance(error, CheckFailure) else type(error).__name__})
            return None

    def request(self, body):
        result = copy.deepcopy(body)
        result["model"] = self.model
        return result

    def chat(self, body):
        response = self.endpoint.chat(self.request(body))
        model = response.get("model") if isinstance(response, dict) else None
        if isinstance(model, str) and len(model) <= 200 and all(c.isalnum() or c in "/-_.:@" for c in model):
            self.observed_models.add(model)
        return validate_response(response, self.expected_response_model)

    def baseline_case(self, baseline, *, stream=False, omit_limit=False):
        body = copy.deepcopy(baseline.request)
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        if omit_limit:
            body.pop("max_tokens", None)
        response = self.chat(body); compare_baseline(response, baseline)
        return summary(response)

    def scopes(self):
        body = copy.deepcopy(self.baselines["text"].request)
        body["chat_template_kwargs"] = {**body.get("chat_template_kwargs", {}), "enable_thinking": True}
        body["max_tokens"] = self.args.scope_max_tokens
        probe = self.chat(body)
        marker = reasoning(probe).lstrip()[:8]
        require(bool(marker), "reasoning probe produced no reasoning; stop scopes remain unverified")
        body["stop"] = [marker]
        explicit = self.chat({**body, "stop_scope": "after_reasoning"})
        default = self.chat(body)
        ordinary = self.chat({**body, "stop_scope": "all"})
        null_default = self.chat({**body, "stop_scope": None})
        require(marker in reasoning(explicit), "after_reasoning stopped inside the initial reasoning")
        require(answer(default) == answer(explicit) and reasoning(default) == reasoning(explicit)
                and default["choices"][0]["finish_reason"] == explicit["choices"][0]["finish_reason"], "server default differs from explicit after_reasoning")
        require(answer(null_default) == answer(default) and reasoning(null_default) == reasoning(default), "null stop_scope differs from the server default")
        require(marker not in reasoning(ordinary) and ordinary["choices"][0]["finish_reason"] == "stop", "explicit all did not apply a reasoning string stop")
        require(ordinary["usage"]["completion_tokens"] < explicit["usage"]["completion_tokens"], "explicit all did not terminate earlier than after_reasoning")
        return {"marker_chars": len(marker), "all": summary(ordinary), "after_reasoning": summary(explicit), "default": summary(default), "null_default": summary(null_default)}

    def heavy_images(self, count):
        body = copy.deepcopy(self.baselines["image"].request)
        parts = [part for m in body["messages"] if isinstance(m.get("content"), list) for part in m["content"]]
        image = next(part for part in parts if part.get("type") == "image_url")
        texts = [part for part in parts if part.get("type") == "text"]
        body.update(messages=[{"role": "user", "content": texts + [copy.deepcopy(image) for _ in range(count)]}], max_tokens=8)
        connection, response = self.endpoint.open("/v1/chat/completions", self.request(body))
        try:
            raw = response.read(LIMIT + 1)
            require(len(raw) <= LIMIT, "image limit response exceeded the byte limit")
            expected = 200 if count == 20 else 400
            detail = {"images": count, "observed_http_status": response.status, "reference_http_status": expected,
                      "reference_limit_match": False, "mode": "observation_only"}
            try:
                result = json.loads(raw)
                if count == 20 and response.status == 200:
                    validate_response(result, self.expected_response_model)
                    detail.update(completion_valid=True, measurements=summary(result), reference_limit_match=True)
                elif count == 21 and response.status == 400:
                    error = result.get("error") if isinstance(result, dict) else None
                    message = error.get("message", "") if isinstance(error, dict) else ""
                    confirmed = isinstance(message, str) and "request has 21 images" in message and "max_images_per_prompt is 20" in message
                    detail.update(count_limit_error_confirmed=confirmed, reference_limit_match=confirmed)
            except (ValueError, TypeError, KeyError):
                detail["valid_reference_response"] = False
            return detail
        finally:
            response.close(); connection.close()

    def models(self):
        status, raw = self.endpoint.get("/v1/models")
        require(status == 200, f"model listing returned HTTP {status}")
        data = json.loads(raw).get("data", [])
        card = next((m for m in data if m.get("id") == self.model), None)
        require(card is not None, "deployed alias is absent from the model list")
        details = {"alias_present": True}
        if self.endpoint is self.metrics:
            declared = card.get("flare", {})
            require(declared.get("default_max_tokens") == 65536, "native discovery does not advertise the requested 65536 default")
            require(declared.get("max_images_per_prompt") == 20, "native discovery does not advertise the requested 20 image limit")
            require(type(declared.get("max_concurrency")) is int and declared["max_concurrency"] >= 4, "native concurrency is below four")
            details.update(default_max_tokens=65536, max_images_per_prompt=20, max_concurrency=declared["max_concurrency"])
        return details

    def run(self):
        self.case("model_alias_listing", self.models)
        for name in ("text", "image"):
            self.case(name + "_temperature_zero", lambda name=name: self.baseline_case(self.baselines[name]))
            self.case(name + "_streaming", lambda name=name: self.baseline_case(self.baselines[name], stream=True))
        self.case("omitted_output_limit", lambda: self.baseline_case(self.baselines["text"], omit_limit=True))
        tool_prompts = {}
        for stream in (False, True):
            for choice in ("none", "auto"):
                def tools(choice=choice, stream=stream):
                    response = self.chat(tools_request(self.baselines["text"], choice, stream=stream))
                    validate_tools(response, choice)
                    tool_prompts[(stream, choice)] = response["usage"]["prompt_tokens"]
                    return summary(response)
                self.case(f"tools_{choice}_{'stream' if stream else 'buffered'}", tools)
        def tool_prefix():
            require(len(tool_prompts) == 4, "tool prefix comparison needs all tool responses")
            require(len(set(tool_prompts.values())) == 1, "tool_choice none changed the shared tool prompt token count")
            return {"prompt_tokens": next(iter(tool_prompts.values()))}
        self.case("shared_tool_prefix", tool_prefix)
        self.case("default_and_explicit_stop_scope", self.scopes)
        self.case("cancellation", lambda: cancelled_request(self.endpoint, self.metrics, self.request(self.baselines["text"].request), wait=self.args.cancel_wait))
        self.case("request_after_cancellation", lambda: self.baseline_case(self.baselines["text"]))
        self.case("repeated_request", lambda: self.baseline_case(self.baselines["text"]))
        def continuation():
            baseline = self.baselines["text"]
            body = copy.deepcopy(baseline.request)
            body["messages"] += [{"role": "assistant", "content": answer(baseline.response)}, {"role": "user", "content": "Repeat your last answer exactly and nothing else."}]
            response = self.chat(body)
            require(answer(response) == answer(baseline.response), "conversation continuation changed the repeated answer")
            return summary(response)
        self.case("conversation_continuation", continuation)
        def concurrent():
            barrier = threading.Barrier(4)
            def one(_):
                barrier.wait(timeout=10)
                response = self.chat(copy.deepcopy(self.baselines["text"].request))
                compare_baseline(response, self.baselines["text"])
                return summary(response)
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(one, range(4)))
            return {"concurrent_http_requests": 4, "responses": responses}
        self.case("concurrency_four", concurrent)
        if self.args.heavy_images:
            for count in (20, 21):
                result = self.case(f"images_{count}_observed", lambda count=count: self.heavy_images(count))
                if result is not None:
                    self.results[-1]["status"] = "observed"
        return {"url": self.endpoint.base, "model": self.model, "expected_response_model": self.expected_response_model, "cases": self.results, "observed_response_models": sorted(self.observed_models),
            "core_passed": all(r["status"] != "failed" for r in self.results),
            "heavy_images_run": self.args.heavy_images,
            "heavy_limit_matches_reference": (all(r["details"]["reference_limit_match"] for r in self.results if r["status"] == "observed") if self.args.heavy_images else None)}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "run"))
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--model", default=ALIAS)
    p.add_argument("--native-url", default="http://127.0.0.1:9999")
    p.add_argument("--gateway-url", default="http://127.0.0.1:8889")
    p.add_argument("--gateway-model", help="explicit gateway request/discovery alias; defaults to --model")
    p.add_argument("--gateway-response-model", help="expected gateway response alias; defaults to upstream --model")
    p.add_argument("--native-key-file", type=Path)
    p.add_argument("--gateway-key-file", type=Path)
    p.add_argument("--require-gateway", action="store_true")
    p.add_argument("--native-only", action="store_true")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--cancel-wait", type=float, default=10)
    p.add_argument("--scope-max-tokens", type=int, default=128)
    p.add_argument("--heavy-images", action="store_true", help="explicitly submit 20/21 image requests and record actual limit behavior")
    p.add_argument("--output", type=Path, help="new JSON result file; existing files are never overwritten")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        require(args.timeout > 0 and args.cancel_wait > 0 and 16 <= args.scope_max_tokens <= 1024, "timeout/cancellation/scope budget is invalid")
        require(args.output is None or not args.output.exists(), "output file must be new")
        gateway_model = args.model if args.gateway_model is None else args.gateway_model
        require(bool(gateway_model) and len(gateway_model) <= 200
                and all(c.isalnum() or c in "/-_.:@" for c in gateway_model), "gateway model alias is invalid")
        gateway_response_model = args.model if args.gateway_response_model is None else args.gateway_response_model
        require(bool(gateway_response_model) and len(gateway_response_model) <= 200
                and all(c.isalnum() or c in "/-_.:@" for c in gateway_response_model), "gateway response model alias is invalid")
        endpoint_url(args.native_url)
        if not args.native_only:
            endpoint_url(args.gateway_url)
        baselines = {name: load_baseline(args.baseline / (name + ".json"), image=name == "image", model=args.model) for name in ("text", "image")}
        report = {"mode": args.action, "executed_http": args.action == "run", "model": args.model,
            "gateway_model": None if args.native_only else gateway_model,
            "gateway_response_model": None if args.native_only else gateway_response_model,
            "baseline": {name: {"images": b.images, "prompt_tokens": b.response["usage"]["prompt_tokens"],
                "completion_tokens": b.response["usage"]["completion_tokens"], "answer_chars": len(answer(b.response))} for name, b in baselines.items()},
            "not_established": ["complete model numerical parity", "DFlash2 performance", "65536 default limit extent"]}
        if args.action == "plan":
            report.update(ready_for_live_run=True, heavy_images_enabled=args.heavy_images, endpoints=[args.native_url] + ([] if args.native_only else [args.gateway_url]))
            exit_code = 0
        else:
            native = Endpoint(args.native_url, args.native_key_file, timeout=args.timeout)
            report["native"] = Suite(native, native, baselines, args).run()
            if not args.native_only:
                gateway = Endpoint(args.gateway_url, args.gateway_key_file, timeout=args.timeout)
                try:
                    status, _ = gateway.get("/v1/models")
                    available = status == 200
                except (OSError, http.client.HTTPException):
                    status, available = None, False
                report["gateway"] = Suite(gateway, native, baselines, args, model=gateway_model,
                    expected_response_model=gateway_response_model).run() if available else {"url": gateway.base, "model": gateway_model,
                    "expected_response_model": gateway_response_model, "status": "unavailable", "probe_http_status": status, "executed_inference": False}
            selected = [report[key] for key in ("native", "gateway") if key in report]
            exit_code = int(any(not item.get("core_passed", not args.require_gateway) or item.get("heavy_limit_matches_reference") is False for item in selected))
        if args.output:
            with args.output.open("x") as output:
                json.dump(report, output, ensure_ascii=False, indent=2); output.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return exit_code
    except Exception as error:
        print(json.dumps({"mode": args.action, "status": "failed", "reason": str(error) if isinstance(error, CheckFailure) else type(error).__name__}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
