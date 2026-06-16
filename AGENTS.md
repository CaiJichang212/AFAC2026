# Project Agent Instructions

本文件约束在 `/Users/lzc/TNTprojectZ/AprojectZ/AFAC2026` 内工作的 agent。除非用户另有明确指示，优先遵守本文件。

## Hard Rules

1. 忽略 `.tmp_del` 目录。不要查看、搜索、读取、依赖或修改该目录中的任何内容。
2. 使用 `uv` 管理 Python 环境和运行命令。优先使用 `uv run ...`、`uv add ...`、`uv sync`，不要直接用系统 Python 安装依赖。
3. 不修改 `open_projects/PageIndex` 源码。PageIndex 作为上游源码依赖使用；项目侧适配、封装、脚本和测试应放在本项目根目录下。
4. 不提交、打印或写入 API Key、密钥、评测标准答案、隐藏测试信息。
5. 生成物、缓存、日志和实验输出应写入 `data/processed_data`、`outputs` 或用户指定目录，不要混入 `data/public_dataset_upload` 原始赛题数据目录。

## Current Task Scope

当前默认工作范围如下：

| 项 | 路径或值 |
| --- | --- |
| PageIndex 源码 | `open_projects/PageIndex` |
| 赛题数据 | `data/public_dataset_upload` |
| 题目文件 | `data/public_dataset_upload/questions/group_a/insurance_questions.json` |
| 文档目录 | `data/public_dataset_upload/raw/insurance` |
| 文档数量 | 16 个 PDF |
| domain | `insurance` |
| split | `A` |
| 默认输出目录 | `outputs/insurance_a` |

除非用户明确要求，不要切换到其他 domain、其他 split 或其他题目文件。

## Data And Ranking Constraints

1. A 榜答题链路只使用题目中给定的 `doc_ids`，不要引入题外文档做召回。
2. 保险题目范围固定为 `data/public_dataset_upload/questions/group_a/insurance_questions.json`。
3. 保险文档范围固定为 `data/public_dataset_upload/raw/insurance` 下的 16 个 PDF。
4. 原始数据目录视为只读。需要中间格式时，生成到 `data/processed_data/...`。
5. 输出文件应保持可审计：答案、证据、模型调用 usage、异常和阶段耗时应分开记录。

## Recommended Workflow

1. 先阅读本文件、相关 `docs/pageindex_finance_gpt5.5pro/` 文档和当前代码，再动手修改。
2. 对数据处理、索引构建、答题流水线等功能变更，优先补充或更新测试。
3. 保持源码通用，不创建只服务单次实验的 `insurance_a` 专用源码目录；domain/split 应通过配置传入。
4. 产物路径建议：
   - 页级缓存：`data/processed_data/pages/insurance`
   - Markdown：`data/processed_data/markdown/insurance`
   - PageIndex 树：`data/processed_data/pageindex/insurance`
   - 文档 catalog：`data/processed_data/catalog/doc_catalog.jsonl`
   - 质量日志：`data/processed_data/quality`
   - 提交答案：`outputs/insurance_a/answer.csv`
   - 证据记录：`outputs/insurance_a/evidence.jsonl`
   - 运行日志：`outputs/insurance_a/logs`
5. 每次生成正式答案后，运行输出校验，至少检查 `answer.csv` 格式、题目覆盖率、答案合法值、证据可追溯性和 usage 日志完整性。

## Command Conventions

1. Python 命令优先使用 `uv run`：
   - `uv run pytest`
2. 若对应脚本已经实现，流水线命令也应通过 `uv run python ...` 执行，例如：
   - `uv run python scripts/build_preprocess.py --domain insurance`
   - `uv run python scripts/build_pageindex.py --domain insurance`
   - `uv run python scripts/run_answers.py --domain insurance --split A`
3. 搜索文件优先使用 `rg` / `rg --files`，并排除 `.tmp_del`。
4. 查看数据结构时可以读取公开赛题 JSON 和保险 PDF 元数据，但不要修改原始数据。

## Verification Expectations

完成代码或流水线变更后，根据变更范围选择验证命令：

1. 配置、解析、计算、schema 变更：运行相关单元测试或 `uv run pytest`。
2. 预处理或索引变更：验证生成物 schema、页码映射、PDF 覆盖率和质量日志。
3. 答题链路变更：验证 `outputs/insurance_a/answer.csv`、`evidence.jsonl`、usage 日志和异常日志。
4. 文档说明变更：至少检查 Markdown 可读性、路径准确性和是否仍遵守当前任务范围。
