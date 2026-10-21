# GLM 5.3 单人推理实验（2026-10-22）

这轮测量保留当前 EXL3/BF16 权重、图片能力、默认思考设置和 max_concurrency=4。所有调整只在隔离监听端口 19999 上试验，生产启动配置保持原值。现有普通解码/DFlash2 输出差异未解决，以下均为诊断测量，不能视为等价推理的提速验收。

## 服务参数对照

归档完成后重新建立基线，同一份原始 DFlash2 TP2 包，固定 4 个公开输入：中文、英文归并排序、长上下文和合成红蓝图片。每组 3 次测量，每次先执行同输入预热，共 24 请求；10 组共 240 请求。全部组的请求计数、生成 token 数、失败计数和两两缓存/工作量检查通过。输入采用 temperature=0、seed=1234；思考选项不覆盖服务默认值。

表中是 native token 时间戳计算的解码吞吐中位数，单位 token/s。前缀已预热，不能把本轮约 6–13 ms TTFT 当成冷启动或首次长输入性能。基线自身也有输出变体，英文 3 个、长上下文 3 个、图片 2 个；结束时原配置复测只有 7/12 输出与初始基线逐 token 相同。

| 配置（最大草稿数/策略） | 中文 | 英文 | 长上下文 | 图片 |
| --- | ---: | ---: | ---: | ---: |
| original-7-balanced | 30.57 | 17.61 | 27.87 | 29.27 |
| draft0-balanced | 18.16 | 18.14 | 18.11 | 18.17 |
| draft2-balanced | 29.05 | 17.62 | 26.76 | 27.94 |
| draft3-balanced | 25.52 | 17.65 | 28.36 | 29.50 |
| draft4-balanced | 30.82 | 17.61 | 26.83 | 32.35 |
| draft2-throughput | 32.42 | 33.39 | 28.44 | 29.98 |
| draft3-throughput | 32.81 | 34.18 | 28.92 | 34.93 |
| draft4-throughput | 34.56 | 36.16 | 29.52 | 29.40 |
| draft7-throughput | 30.03 | 30.91 | 26.13 | 27.16 |
| closing-control | 31.55 | 17.54 | 28.65 | 30.43 |

`throughput` 和草稿上限 3/4 值得继续复测，但各任务收益不同；单独降低草稿上限没有普遍收益。每个候选配置都有输出不一致，不能据英文吞吐约翻倍就声称已获得正确提速。最终仍使用原生产配置。

原始记录在 spark-01 `/workspace/outputs/flare-beiyu/perf-20261022-attempt01/`：每组 `results.json`、对应 `*-configuration.json`、`*-comparison.json`。比较包含输出是否相等、首个差异位置及样本缓存条件；不会打印回答正文。

## 词表通信和候选合并

对实际 TP2 head 权重、8 行 BF16 隐藏向量进行组件测量。每个路径捕获 CUDA graph，通过 native worker 重放，每项 7 组 × 32 次，检查 GPU 分配计数不变，并保留 CUPTI 核函数时间线。

- 全词表 all-gather：每 rank 1,239,040 字节，约 801.89 微秒。
- 本地 top-16 后交换候选：每 rank 768 字节；单独交换约 13.34 微秒，包含候选合并约 62.56 微秒。
- head+候选路径测得约 3.94 ms → 2.57 ms。这是组件测量，尚未集成导出模型，不能推算 HTTP 提速比例。
- 6 个正确性输入中有 3 个候选集合/分数映射不同，含 2 个实际权重输入及截断边界并列分数。分数数值相等并不代表候选 token 集合相等；两阶段 top-k 的并列处理未通过原路径语义检查，当前原型不能上线。

记录：`communication-rank0.json`、rank1 对应文件及两台机器的 `.trace.json`。目前原型只测单请求最宽的 8 行，尚未验证完整图容量、采样参数、状态转移及最终服务。

## 两处分支并行

两台 Spark 的 KDA 门控投影链、EXL3 routed experts 与 BF16 shared expert 分别测了 1/4/8 行。先修复测试脚本的参数顺序和遗漏的 ScratchArena 注册，再按现有组件的完整准备/绑定流程重跑。成功记录在两台机器的 `/workspace/outputs/flare-beiyu/perf-20261022-branches-03/`；此前失败日志保留。

每台 15 条测量，共 30 条，执行前及 native 重放后与串行结果精确一致，GPU 分配检查通过，12 份 CUPTI 时间线都有实际核函数记录。分支测量是各 rank 本地耗时，没有把它标成跨 rank 同步的完整模型耗时。

- KDA 组件耗时下降约 0.8%–2.9%。
- 专家晚发并行下降约 0.2%–1.5%；早发并行在部分配置下反而回退约 1.9%。
- 差异较小，尚未验证完整 forward 编译、所有图容量、共享 scratch 交替、状态边界或导出包的服务收益。本轮不接入生产，也不把组件百分比当作整模型加速。

`qualify/glm_gpu_optimizations.py` 保存这两种诊断：`communication` 需两个 rank 同时运行，`branches` 分别运行。显式传入 `--rank`、当前 `--checkpoint`、已验证的 `--exl3-library` 和全新 `--output` 路径；通信还可指定 `--rendezvous`。必须在 GPU 空闲时测，启动/恢复服务由运维控制，不由该测量脚本操作。

## 重复使用基线脚本

`qualify/glm_service_benchmark.py` 仅使用 Python 标准库；每次输出目录必须是新目录，不覆盖旧结果。API key 通过文件读取，不写入结果。结果保留 token ID 用于精确比较，文件权限 0600，不保存回答正文。应选择隔离服务入口，或先确保没有其他请求；脚本发现计数干扰会把结果标为无效。

```sh
python3 -m qualify.glm_service_benchmark corpus --output /tmp/glm-corpus.json
python3 -m qualify.glm_service_benchmark run --corpus /tmp/glm-corpus.json \
  --base http://127.0.0.1:19999 --label baseline --configuration /tmp/baseline-config.json \
  --repeats 3 --output /tmp/glm-baseline
# 用相同 corpus、repeats 和缓存条件测试调整后的配置，输出到另一个新目录。
python3 -m qualify.glm_service_benchmark compare \
  --baseline /tmp/glm-baseline/results.json --candidate /tmp/glm-candidate/results.json \
  --output /tmp/glm-comparison.json
```

`--configuration` 记录人工提供的实际配置，不会修改服务；使用前应填写权重包、设备、启动参数和思考设置。NewAPI 可通过 `--api-key-file` 和指向 native 的 `--metrics-base` 测试；计时/ID 扩展必须完整透传，否则拒绝测量。

脚本的 4 项真实 HTTP/SSE CPU 测试通过：多 token 分组使用 native 时间、缺失时间戳拒绝、等待 SSE 后请求退休、额外请求使隔离失败。完整模型数值问题继续追踪于 [DFlash2 输出差异](glm53-dflash2-output-divergence.md)。
