# GLM service acceptance

`qualify.glm_service_acceptance` is a standard-library HTTP client, with no model
or CUDA imports. Run it on spark-01, where the original text/image baseline files
already reside. It reads the original request/response envelopes in place. Saved
results contain counts, timings and check outcomes, never prompts, image bytes,
answer text or credential values. It does not alter services, accounts or keys.

## Check inputs without inference

From `/workspace/projects/flare-beiyu` on spark-01:

```sh
python3 -m qualify.glm_service_acceptance plan \
  --baseline /workspace/outputs/flare-beiyu/baseline
```

`plan` validates the real envelope structure, temperature-zero request, model
alias, embedded PNG header and token accounting. It performs no HTTP requests and
produces no service pass claims. Offline unit tests use separate synthetic inputs:

```sh
.venv/bin/python -m pytest tests/reference/test_glm_service_acceptance.py -q
```

## Check the deployed native service and NewAPI

First load the prepared native package using the two-host task launcher. The
service must use alias `GLM-5.3-Flash-EXL3`, default stop scope `after_reasoning`,
default output budget 65536, maximum image count 20, and concurrency at least four.
Native `--max-images-per-prompt 20` sets the actual shared processor count limit;
recipe image planning alone does not enforce that service policy.

After the native 9999 listener is healthy:

```sh
python3 -m qualify.glm_service_acceptance run \
  --baseline /workspace/outputs/flare-beiyu/baseline \
  --native-key-file /workspace/runtime/flare-beiyu/api-key \
  --gateway-url http://127.0.0.1:8889 \
  --gateway-key-file /workspace/runtime/flare-beiyu/api-gateway-key \
  --gateway-model spark/GLM-5.3-Flash-EXL3 \
  --gateway-response-model GLM-5.3-Flash-EXL3 \
  --require-gateway \
  --output /workspace/outputs/flare-beiyu/api-acceptance-new.json
```

Port 8889 is the confirmed daily NewAPI entry point. Its enabled channel exposes
`spark/GLM-5.3-Flash-EXL3`. Its raw passthrough preserves extra fields such as
`stop_scope` and also preserves that request alias. Native 9999 must accept both
`GLM-5.3-Flash-EXL3` and `spark/GLM-5.3-Flash-EXL3`, in that order, so its responses
keep the first alias. The command uses the existing PUBLIC credential
in the task's mode-600 key file. It changes no credential, channel or nginx
configuration. Port 8888 is outside this acceptance path; its earlier compatibility
proposal was not applied.

`--gateway-model` selects the gateway request and discovery alias. The expected
gateway response alias defaults to upstream `--model`; use
`--gateway-response-model` only when the deployed gateway has a different known
response contract. The current NewAPI v1.0.0-rc.41 preserves the upstream model
name in both buffered OpenAI responses and SSE, including synthesized usage
frames. Its [buffered handler](https://github.com/QuantumNous/new-api/blob/v1.0.0-rc.41/relay/channel/openai/relay-openai.go#L251-L366),
[stream forwarding](https://github.com/QuantumNous/new-api/blob/v1.0.0-rc.41/relay/channel/openai/relay-openai.go#L22-L39)
and [final stream usage handling](https://github.com/QuantumNous/new-api/blob/v1.0.0-rc.41/relay/channel/openai/helper.go#L165-L204)
do not replace it with the prefixed request alias. The native suite continues to
require `--model` in discovery and every response. The result records the request
`model`, `expected_response_model`, and observed response model names separately.
Any other response alias fails; answer, reasoning and token comparisons remain
exact. `--gateway-model` defaults to `--model` when omitted.

Blank key files omit bearer authentication, matching the current native upstream
configuration. Keys cannot be embedded in URLs. Endpoint URLs may name the server
root or `/v1`; requests and native metrics use the appropriate paths. The output
file must be new.

The suite checks discovery and response aliases, advertised native limits,
temperature-zero text and PNG responses, streamed text/reasoning and usage,
`tool_choice` none/auto with the same tool prefix, omitted/null/default and explicit
string-stop scopes, client disconnect cancellation through native metrics,
subsequent/repeated/conversation requests, and four concurrent HTTP requests.
It compares baseline answers, reasoning, token counts, finish reasons and image
token costs exactly. A difference fails the case; the tool does not soften the
reference to obtain a passing result. Tool auto must actually return the requested
call. The reasoning-scope check must observe real reasoning or report failure.

Without `--require-gateway`, an unavailable gateway is reported separately. A
reachable gateway still executes all checks and must pass them. Use
`--native-only` for an explicit native-only run. `--timeout` controls each HTTP
request, `--cancel-wait` the cancellation-metric deadline, and
`--scope-max-tokens` the bounded reasoning probe (default 128).

## Explicit image-count workload

Append `--heavy-images` to submit repeated copies of the baseline PNG in 20-image
and 21-image prompts. This performs real image inference for the 20-image case.
The result separately records actual HTTP status and whether it matches the
reference policy: valid completion for 20, and a count-limit error identifying
21 images against limit 20 for the rejected request. A different HTTP 400, such as
a body or context limit, does not qualify the image count policy. A mismatch
returns nonzero. These results stay marked as observations so an unimplemented
or misconfigured limit cannot be presented as an existing guarantee.

Exit 0 means the selected completed checks passed, 1 means a live check or image
limit observation failed, and 2 means input/setup validation failed. Partial case
failures remain in the JSON result. Complete model numerical parity, DFlash2
performance, and the full 65536-token output extent require separate workloads;
the short API checks do not establish them. No live service evidence is produced
until `run` has actually contacted the deployed service.
