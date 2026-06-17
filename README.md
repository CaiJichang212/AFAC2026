# AFAC2026 A榜保险 PageIndex 基线

本仓库实现面向 `insurance` / `A` split 的 PageIndex 可复现流水线。PageIndex 源码作为上游依赖保留在 `open_projects/PageIndex`，项目侧只通过 `agent/pageindex_adapter.py` 调用底层 `md_to_tree()` / `page_index()`。

## 范围

- 题目：`data/public_dataset_upload/questions/group_a/insurance_questions.json`
- PDF：`data/public_dataset_upload/raw/insurance` 下 16 个文件
- 默认输出：`outputs/insurance_a`
- 中间产物：`data/processed_data`

当前实现提供规则化可审计基线：离线构建页缓存、Markdown、PageIndex 树、node spans 和 catalog；在线链路按题目给定 `doc_ids` 读取候选节点、页段证据、计算与答案格式化。`agent/llm_client.py` 已保留 Qwen 兼容 JSON 调用入口，正式接入外部模型时所有调用应通过该入口和 `TokenMeter` 统计 usage。

## 依赖

使用 `uv` 运行命令：

```bash
uv sync
```

正式 Qwen 调用应配置赛题允许的 Qwen 模型和对应 API Key。`docs/模型配置.md` 中的 `ark-code-latest` 仅用于开发调试，不作为正式提交链路的推理结果来源。

## 复现

```bash
uv run python scripts/build_preprocess.py --domain insurance --split A
uv run python scripts/build_pageindex.py --domain insurance --split A
uv run python scripts/build_catalog.py --domain insurance --split A
uv run python scripts/run_answers.py --domain insurance --split A
uv run python scripts/validate_outputs.py --domain insurance --split A
```

生成文件包括：

- `outputs/insurance_a/answer.csv`
- `outputs/insurance_a/evidence.jsonl`
- `outputs/insurance_a/logs/usage.jsonl`

## 验证

```bash
uv run pytest tests/agent -q
```

## 合规边界

- 不读取 `.tmp_del`
- 不修改 `open_projects/PageIndex`
- A 榜链路只使用题目中的 `doc_ids`
- 原始赛题数据目录保持只读
- API Key、标准答案、隐藏测试信息不写入日志或产物
