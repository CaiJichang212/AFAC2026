# A榜保险 PageIndex 开发计划文档

## Summary

**目标：** 基于 `docs/pageindex_finance_gpt5.5pro/A榜保险PageIndex系统架构设计.md`，实现可复用到其他题目和 B 榜的 PageIndex 问答流水线；本阶段运行范围固定为 `domain=insurance`、`split=A`、20 道 `insurance_questions.json`、16 个保险 PDF。

**关键事实：** 保险数据共 16 个 PDF、299 页、文本层完整；但 12 个 PDF 无可靠 outline，适合“页级文本缓存 + Markdown 主索引 + PDF 兜底”。题库 20 题，其中 `mcq=7`、`multi=13`，每题给定 2-4 个 `doc_ids`，A 榜无需候选文档召回。

**计划文档落点：** 执行时新增 `docs/pageindex_finance_gpt5.5pro/A榜保险PageIndex开发计划.md`。源码路径使用通用目录，不出现 `agent/insurance_a/` 或 `scripts/insurance_a/` 这类榜单专用代码路径。

## Architecture

新增项目侧实现，不修改 `open_projects/PageIndex`：

- `agent/config.py`：集中定义 `domain/split/raw_dir/questions_path/output_dir/model`，正式推理模型必须显式配置；`docs/模型配置.md` 的 ARK 配置只作为本地开发调试默认项。
- `agent/domain_profiles/insurance.py`：保险领域关键词、产品别名、题型提示、计算类型和质量阈值；后续其他领域新增同级 profile。
- `agent/pageindex_adapter.py`：薄封装 `pageindex.page_index_md.md_to_tree()` 和 `pageindex.page_index.page_index()`，显式关闭 `doc_description/node_text`，默认不生成摘要。
- `agent/preprocess.py`：生成页级缓存、结构化 Markdown、`page_map`、解析质量日志。
- `agent/index_store.py`：统一读取 Markdown 主索引、PDF 兜底索引和页缓存，向后续模块只暴露 PDF 页码范围。
- `agent/question_parser.py`、`doc_retriever.py`、`tree_retriever.py`、`evidence_extractor.py`、`calculation.py`、`answer_judge.py`：通用问答链路，保险差异通过 `domain_profiles/insurance.py` 注入。
- `agent/llm_client.py`、`token_meter.py`：统一 JSON 调用、重试、usage 记录、阶段耗时，避免各模块直接调用模型。
- `scripts/build_preprocess.py`、`build_pageindex.py`、`run_answers.py`、`validate_outputs.py`：全部通过 `--domain insurance --split A` 选择任务。
- `tests/agent/`：schema、页码映射、索引适配、答案合法化、端到端 smoke 测试。

## Interfaces And Contracts

核心数据产物仍按领域和榜单隔离：

- `data/processed_data/pages/{domain}/{doc_id}.jsonl`
- `data/processed_data/markdown/{domain}/{doc_id}.md`
- `data/processed_data/markdown/{domain}/{doc_id}.page_map.json`
- `data/processed_data/pageindex/{domain}/{doc_id}.json`
- `data/processed_data/pageindex/{domain}/{doc_id}.pdf_fallback.json`
- `data/processed_data/pageindex/{domain}/{doc_id}.node_spans.json`
- `data/processed_data/catalog/doc_catalog.jsonl`
- `outputs/{domain}_{split}/answer.csv`
- `outputs/{domain}_{split}/evidence.jsonl`
- `outputs/{domain}_{split}/logs/*.jsonl`

公开接口必须稳定：

```python
QuestionParser.parse(raw_question: dict, profile: DomainProfile) -> ParsedQuestion
DocRetriever.retrieve(parsed: ParsedQuestion, catalog: DocCatalog) -> list[str]
IndexStore.get_document_metadata(doc_id: str) -> dict
IndexStore.get_document_structure(doc_id: str) -> list[dict]
IndexStore.get_page_content(doc_id: str, pages: str) -> list[PageText]
TreeRetriever.retrieve(parsed: ParsedQuestion, doc_id: str, compact_tree: list[dict]) -> list[CandidateNode]
EvidenceExtractor.extract(parsed: ParsedQuestion, candidates: list[CandidateNode]) -> list[EvidenceRecord]
CalculationEngine.compute(parsed: ParsedQuestion, evidence: list[EvidenceRecord]) -> list[CalculationRecord]
AnswerJudge.judge(parsed: ParsedQuestion, evidence: list[EvidenceRecord], calculations: list[CalculationRecord]) -> AnswerRecord
```

PageIndex 调用强约束：

```python
await md_to_tree(
    md_path=md_path,
    if_add_node_summary="no",
    if_add_doc_description="no",
    if_add_node_text="no",
    if_add_node_id="yes",
    model=config.pageindex_model,
)
```

`IndexStore.get_document_structure()` 返回紧凑树时必须删除 `text` 字段；Markdown 的 `line_num` 只能通过 `node_spans` 转成 `page_range`，不得传给问答模块。

## Implementation Plan

1. 撰写开发计划文档，固定范围、目录、接口契约、产物契约、验收标准，并注明不修改 `open_projects/PageIndex`。
2. 建立通用 `agent/` 包和 `scripts/` 入口；配置读取支持 CLI 参数和环境变量，当前默认运行 `--domain insurance --split A`。
3. 实现 `DomainProfile` 与 `domain_profiles/insurance.py`，把保险关键词、产品别名、计算类型和质量阈值从通用链路中分离。
4. 实现预处理：用 PyMuPDF 抽取 16 个 PDF 页文本，生成页缓存、基础 Markdown、`page_map` 和质量日志。
5. 实现 PageIndex 适配：Markdown 主路调用 `md_to_tree()`；PDF 兜底调用 `page_index()`；全部显式传参，不使用 `PageIndexClient.index()` 和默认 `config.yaml`。
6. 实现 `node_spans`：根据 Markdown 标题行、下一个同级/更高层级标题、`page_map` 生成 `source_page_range`，并做页码边界校验。
7. 实现 `doc_catalog`：从 PDF 文件名、解析出的标题、题目中产品名/别名信号生成保险文档元数据；A 榜仅用于校验和检索增强，B 榜可扩展为候选召回输入。
8. 实现在线流水线：题目解析、A 榜 `doc_ids` passthrough、结构树节点选择、页段取原文、选项级证据抽取、计算题程序化处理、答案合法化。
9. 实现输出与审计：`answer.csv` 严格 20 行；`evidence.jsonl` 保留证据字段；日志记录模型、stage、token、耗时、异常。
10. 加入质量回退：Markdown 索引节点太少、页码缺失、关键标题覆盖不足时启用 PDF 兜底；两者失败则降级页级关键词检索并写 warning。

## Test And Acceptance

测试命令建议：

- `pytest tests/agent -q`
- `python scripts/build_preprocess.py --domain insurance --split A`
- `python scripts/build_pageindex.py --domain insurance --split A`
- `python scripts/run_answers.py --domain insurance --split A`
- `python scripts/validate_outputs.py --domain insurance --split A`

验收标准：

- 不读取或依赖 `.tmp_del`。
- 不修改 `open_projects/PageIndex`。
- 代码路径不使用 `insurance_a` 专用目录；领域差异只放入 `agent/domain_profiles/insurance.py` 和数据/输出路径参数。
- `answer.csv` 包含全部 20 个 `qid`，答案只含合法选项字母；`mcq` 必须单字母，`multi` 必须去重排序。
- A 榜在线检索只使用题目给定 `doc_ids`，不得引入题外文档。
- 紧凑树不含节点正文 `text`，证据判断必须来自页级原文。
- 每个最终选中选项至少有一条 `support` 证据；未能确定时必须记录 `unclear` 和补检索日志。
- 所有模型调用都有 stage、model、prompt/completion token、耗时和错误记录。
- 至少覆盖以下测试场景：Markdown 行号到页码映射、PDF 兜底页码归一化、空/坏节点质量判定、多选答案合法化、计算题金额/比例归一化、A 榜 doc_ids passthrough。

## Assumptions

- 首期运行目标仍是 A 榜保险，但代码结构按多领域、多 split 设计。
- OCR 不是首期主链路，因为 16 个保险 PDF 当前每页都有可抽取文本；保留 OCR 兜底接口即可。
- ARK `ark-code-latest` 仅作为本地开发调试配置；正式赛题推理模型通过配置显式指定为允许的 Qwen 系列，并纳入 token 统计。
- PageIndex 索引构建视为离线阶段；若 PDF 兜底索引调用 LLM，也必须记录构建日志，但不混入每题在线答题 token。
