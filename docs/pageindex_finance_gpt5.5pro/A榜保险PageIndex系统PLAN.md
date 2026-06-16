# A榜保险 PageIndex 系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改 `open_projects/PageIndex` 源码的前提下，构建面向 A 榜保险条款题目的可复现 PageIndex 问答流水线，输出满足赛题格式的 `answer.csv`、可审计的 `evidence.jsonl` 和完整 Token/日志记录。

**Architecture:** 系统采用“离线解析与结构索引 + 在线结构检索与证据推理”的确定性流水线。PageIndex 只承担单文档内部树索引能力，项目侧负责题目解析、A 榜 `doc_ids` passthrough、节点选择、页段读取、选项级证据抽取、程序化计算、答案合法化和审计输出。

**Tech Stack:** Python 3、PyMuPDF/fitz、PageIndex local source、Qwen API compatible client、pytest、JSONL/CSV artifacts。

---

## 1. Scope And Ground Truth

当前计划只覆盖以下范围：

| 项 | 值 |
| --- | --- |
| PageIndex 源码 | `open_projects/PageIndex` |
| 题目文件 | `data/public_dataset_upload/questions/group_a/insurance_questions.json` |
| 文档目录 | `data/public_dataset_upload/raw/insurance` |
| domain | `insurance` |
| split | `A` |
| 输出目录 | `outputs/{domain}_{split_lower}`，当前为 `outputs/insurance_a` |

实测数据特征：

| 指标 | 值 |
| --- | --- |
| PDF 数量 | 16 |
| PDF 总页数 | 299 |
| 可抽取文本页 | 299 / 299 |
| 总字符数 | 329177 |
| 无内嵌 outline 文档 | 13 / 16 |
| 题目数量 | 20 |
| 单选题 | 7 |
| 多选题 | 13 |
| 每题选项数 | 4 |
| 每题 doc_ids 数 | 2 到 4 |

赛题强约束：

- 不读取、不依赖 `.tmp_del`。
- 不修改 `open_projects/PageIndex`。
- A 榜在线链路只使用题目给定 `doc_ids`，不引入题外文档。
- 推理问答阶段的文档定位、检索、压缩、证据判断、答案生成和自检必须使用 Qwen 系列模型并统计 Token。
- `docs/模型配置.md` 中的 `ark-code-latest` 只作为开发调试配置；其结果不得进入正式提交链路。
- `PageIndexClient.index()` 不是正式入口，因为它会生成随机 UUID、默认开启摘要/正文/文档描述；PageIndex 源码会读取默认 `config.yaml`，但项目侧必须显式覆盖模型和摘要/正文/描述等赛题相关参数。

## 2. Target Architecture

```text
raw insurance PDFs
  -> preprocess pages + markdown + page_map
  -> PageIndex markdown tree + node_spans
  -> optional PageIndex PDF fallback
  -> doc catalog + quality reports
  -> question parser
  -> A split doc_ids passthrough
  -> compact tree node retrieval
  -> page text evidence extraction
  -> calculation + answer judging
  -> answer.csv + evidence.jsonl + logs
```

分层职责：

| 层 | 责任 | 禁止事项 |
| --- | --- | --- |
| 文档解析层 | 抽取页文本、生成 Markdown、生成 `page_map`、记录解析质量 | 不生成可直接用于正式答题的非 Qwen 语义摘要 |
| PageIndex 结构索引层 | 调用 `md_to_tree()` 和必要时 `page_index()` 生成树 | 不调用 `PageIndexClient.index()`；不修改 PageIndex |
| 索引存储层 | 屏蔽 Markdown 行号和 PDF 页码差异，只暴露 PDF 页码范围 | 不向问答模块暴露 Markdown `line_num` |
| 题目与检索层 | 解析题干/选项/产品/数值；A 榜返回题目 `doc_ids` | A 榜不做候选文档召回 |
| 证据记忆层 | 按选项抽取 `support/refute/unclear` 证据 | 不用树摘要直接判断答案 |
| 推理与计算层 | 程序化处理金额、比例、排序、扣减；合法化答案 | 不直接写模型原始输出 |
| 输出审计层 | 写 `answer.csv`、`evidence.jsonl`、usage、阶段日志 | 不保存 API Key 或标准答案 |

## 3. Repository File Plan

源码路径保持通用，不创建 `agent/insurance_a/` 或 `scripts/insurance_a/` 这类榜单专用目录。

### 3.1 Create

| 路径 | 责任 |
| --- | --- |
| `agent/__init__.py` | 标记通用 Agent 包。 |
| `agent/config.py` | 读取 CLI/env 配置，统一 domain、split、路径、模型、预算。 |
| `agent/schemas.py` | 定义 `ParsedQuestion`、`CandidateNode`、`EvidenceRecord`、`AnswerRecord` 等数据契约。 |
| `agent/domain_profiles/__init__.py` | 领域 profile 注册入口。 |
| `agent/domain_profiles/insurance.py` | 保险关键词、产品别名、责任类别、计算类型、质量阈值。 |
| `agent/preprocess.py` | PDF 页文本、Markdown、page_map、解析质量产物。 |
| `agent/pageindex_adapter.py` | 项目侧薄封装 PageIndex `md_to_tree()` / `page_index()`。 |
| `agent/index_store.py` | 统一读取树、node_spans、页缓存和 catalog。 |
| `agent/catalog.py` | 构建和读取 `doc_catalog.jsonl`。 |
| `agent/question_parser.py` | 解析题目字段、选项信号、产品别名、数值条件。 |
| `agent/doc_retriever.py` | A 榜 passthrough；保留 B 榜候选召回接口。 |
| `agent/tree_retriever.py` | 基于紧凑 PageIndex 树选择候选节点。 |
| `agent/evidence_extractor.py` | 读取页段原文，按选项抽取证据记忆。 |
| `agent/calculation.py` | 金额、比例、排序、扣减的程序化计算。 |
| `agent/answer_judge.py` | 逐选项判断和答案格式合法化。 |
| `agent/llm_client.py` | Qwen 兼容 API 调用、JSON schema 校验、重试和错误包装。 |
| `agent/token_meter.py` | usage 聚合、阶段耗时、日志写入。 |
| `agent/pipeline.py` | 编排单题与全量题目运行。 |
| `scripts/build_preprocess.py` | CLI：生成页缓存、Markdown、page_map、parse quality。 |
| `scripts/build_pageindex.py` | CLI：生成 PageIndex 主索引、兜底索引、node_spans、index quality。 |
| `scripts/build_catalog.py` | CLI：生成 `doc_catalog.jsonl`。 |
| `scripts/run_answers.py` | CLI：运行 A 榜保险答题流水线。 |
| `scripts/validate_outputs.py` | CLI：校验输出 schema、答案格式、证据追溯和 usage。 |
| `tests/agent/test_config.py` | 配置路径和默认值测试。 |
| `tests/agent/test_domain_profiles.py` | 保险领域 profile、关键词和别名测试。 |
| `tests/agent/test_preprocess.py` | 页缓存、Markdown、page_map schema 测试。 |
| `tests/agent/test_pageindex_adapter.py` | PageIndex 显式参数和禁用高层入口测试。 |
| `tests/agent/test_index_store.py` | `line_num -> page_range`、PDF 兜底归一化测试。 |
| `tests/agent/test_catalog.py` | catalog 覆盖率和字段 schema 测试。 |
| `tests/agent/test_question_parser.py` | 产品别名、数值条件、选项关键词解析测试。 |
| `tests/agent/test_doc_retriever.py` | A 榜 `doc_ids` passthrough 和题外文档禁止测试。 |
| `tests/agent/test_calculation.py` | 金额、比例、扣减和排序计算测试。 |
| `tests/agent/test_answer_judge.py` | `mcq/multi/tf` 答案合法化测试。 |
| `tests/agent/test_validate_outputs.py` | `answer.csv`、`evidence.jsonl` 验收测试。 |

### 3.2 Modify

| 路径 | 修改 |
| --- | --- |
| `requirements.txt` | 如果项目根不存在，新增项目依赖；不要改 PageIndex 自带 `requirements.txt`。 |
| `README.md` | 如果项目根不存在，新增一键运行说明；不覆盖 PageIndex README。 |
| `docs/pageindex_finance_gpt5.5pro/A榜保险PageIndex系统PLAN.md` | 本计划文档。 |

### 3.3 Generated Artifacts

| 路径 | 说明 |
| --- | --- |
| `data/processed_data/pages/{domain}/{doc_id}.jsonl` | 页级原文缓存，每行一页。 |
| `data/processed_data/markdown/{domain}/{doc_id}.md` | 带标题层级的 Markdown。 |
| `data/processed_data/markdown/{domain}/{doc_id}.page_map.json` | Markdown 行号到 PDF 页码映射。 |
| `data/processed_data/pageindex/{domain}/{doc_id}.json` | Markdown 主路 PageIndex 树。 |
| `data/processed_data/pageindex/{domain}/{doc_id}.pdf_fallback.json` | PDF 兜底 PageIndex 树，仅质量触发或实验全量生成。 |
| `data/processed_data/pageindex/{domain}/{doc_id}.node_spans.json` | `node_id -> source_page_range`。 |
| `data/processed_data/catalog/doc_catalog.jsonl` | 文档元数据和产品别名。 |
| `data/processed_data/quality/{domain}_parse_quality.jsonl` | 解析质量日志。 |
| `data/processed_data/quality/{domain}_index_quality.jsonl` | 索引质量日志。 |
| `outputs/{domain}_{split_lower}/answer.csv` | 赛题提交文件。 |
| `outputs/{domain}_{split_lower}/evidence.jsonl` | 证据、计算、自检、回退记录。 |
| `outputs/{domain}_{split_lower}/logs/*.jsonl` | LLM usage、异常、阶段耗时。 |

## 4. Interface Contracts

### 4.1 Config

`AgentConfig` 必须由 CLI 参数和环境变量构造，默认只覆盖当前任务。

```python
@dataclass(frozen=True)
class AgentConfig:
    domain: str = "insurance"
    split: str = "A"
    raw_root: Path = Path("data/public_dataset_upload/raw")
    questions_root: Path = Path("data/public_dataset_upload/questions")
    processed_root: Path = Path("data/processed_data")
    pageindex_root: Path = Path("open_projects/PageIndex")
    output_root: Path = Path("outputs")
    inference_model: str = "dashscope/qwen3.6-plus"
    dev_model: str | None = None
    toc_check_page_num: int = 20
    max_page_num_each_node: int = 8
    max_token_num_each_node: int = 20000
    max_docs_per_question: int = 4
    max_nodes_per_doc: int = 5
    max_pages_per_doc: int = 8
    max_evidence_per_option: int = 3
    max_retry_per_question: int = 1
```

派生路径规则：

| 字段 | 当前值 |
| --- | --- |
| `raw_dir` | `data/public_dataset_upload/raw/insurance` |
| `questions_path` | `data/public_dataset_upload/questions/group_a/insurance_questions.json` |
| `output_dir` | `outputs/insurance_a` |
| `pages_dir` | `data/processed_data/pages/insurance` |
| `pageindex_dir` | `data/processed_data/pageindex/insurance` |
| `logs_dir` | `outputs/insurance_a/logs` |

PageIndex 默认配置处理：

- PageIndex `page_index()` 内部会通过 `ConfigLoader` 读取 `open_projects/PageIndex/pageindex/config.yaml`，再合并项目侧传入参数。
- 项目侧不得依赖默认 `model`、`if_add_node_summary`、`if_add_doc_description`、`if_add_node_text`、`toc_check_page_num`、`max_page_num_each_node`、`max_token_num_each_node` 等值。
- 适配器必须显式覆盖赛题相关参数，并把实际生效配置写入构建日志。
- `max_docs_per_question`、`max_nodes_per_doc`、`max_pages_per_doc`、`max_evidence_per_option`、`max_retry_per_question` 是在线检索预算配置，不允许在检索代码中散落硬编码常量。

### 4.2 PageIndex Adapter

PageIndex 只通过项目侧适配器调用。适配器负责插入 `open_projects/PageIndex` 到 `sys.path`，传入显式参数，保存赛题 `doc_id`，并把模型、开关和日志落盘。

关键调用契约：

```python
result = await md_to_tree(
    md_path=str(md_path),
    if_add_node_summary="no",
    if_add_doc_description="no",
    if_add_node_text="no",
    if_add_node_id="yes",
    model=config.inference_model,
)
```

PDF 兜底必须显式传参：

```python
result = page_index(
    doc=str(pdf_path),
    model=config.inference_model,
    toc_check_page_num=config.toc_check_page_num,
    max_page_num_each_node=config.max_page_num_each_node,
    max_token_num_each_node=config.max_token_num_each_node,
    if_add_node_summary="no",
    if_add_doc_description="no",
    if_add_node_text="no",
    if_add_node_id="yes",
)
```

禁止项测试必须覆盖：

- 不调用 `PageIndexClient.index()`。
- 不依赖 `open_projects/PageIndex/pageindex/config.yaml` 的默认模型、摘要、正文、文档描述和 PDF 构树预算参数。
- 不把 `text` 字段写入紧凑树给在线问答模块。

### 4.3 Data Schemas

页缓存每行：

```json
{"doc_id":"1","page":1,"text":"...","char_count":1234,"source_path":"data/public_dataset_upload/raw/insurance/1.pdf"}
```

`page_map`：

```json
{
  "doc_id": "1",
  "markdown_path": "data/processed_data/markdown/insurance/1.md",
  "line_to_page": {"1": 1, "2": 1, "83": 2}
}
```

`node_spans` 每条：

```json
{
  "doc_id": "1",
  "node_id": "0001",
  "title": "身故保险金",
  "start_line": 120,
  "end_line": 176,
  "start_page": 6,
  "end_page": 8,
  "source_page_range": "6-8"
}
```

`CandidateNode`：

```json
{
  "doc_id": "1",
  "node_id": "0001",
  "title": "身故保险金",
  "page_range": "6-8",
  "matched_signals": ["身故保险金", "账户价值"],
  "reason": "标题和题干责任类别匹配",
  "needs_page_fetch": true
}
```

`EvidenceRecord`：

```json
{
  "qid": "ins_a_001",
  "doc_id": "1",
  "node_id": "0001",
  "pages": "6-8",
  "option": "B",
  "evidence_type": "support",
  "quote": "身故保险金按保单账户价值给付",
  "normalized_fact": "领取日前身故保险金=保单账户价值",
  "numbers": [{"name": "保单账户价值", "value": 900000, "unit": "元"}],
  "confidence": "high"
}
```

`answer.csv`：

| qid | answer | prompt_tokens | completion_tokens | total_tokens |
| --- | --- | --- | --- | --- |
| summary |  | 0 | 0 | 0 |
| ins_a_001 | B | 0 | 0 | 0 |

`summary` 必须是第一行。题目行必须覆盖 20 个 `qid`。

### 4.4 Public Python Interfaces

接口名稳定，领域差异通过 `DomainProfile` 注入。

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

`IndexStore.get_document_structure()` 返回字段白名单：

- `node_id`
- `title`
- `summary` optional / nullable
- `prefix_summary` optional / nullable
- `page_range`
- `nodes`
- `index_source`

即使底层索引含 `line_num`、`start_index`、`end_index` 或 `text`，在线问答模块也只能看到统一后的 `page_range`。当前默认主路关闭 `if_add_node_summary`，因此树检索不得把空的 `summary` 或 `prefix_summary` 当作检索依据；默认检索信号是 `title`、`page_range`、层级关系、保险关键词、题干和选项信号。若未来启用 summary，必须使用正式 Qwen 模型生成并记录构建 Token。

### 4.5 LLM Usage And PageIndex Build Tokens

在线推理问答阶段的 Qwen 调用必须通过 `LLMClient` 和 `TokenMeter`，调用日志至少包含 `qid`、`stage`、`model`、`prompt_tokens`、`completion_tokens`、`total_tokens`、`latency_ms`、`success`、`error`。

PageIndex 离线索引构建的 LLM 调用单独写入构建日志，不混入每题在线答题 Token。若正式答题阶段触发 PageIndex PDF 兜底构建、索引修复或摘要生成，则这些调用必须纳入对应题目的 Token 统计。实现时需要通过项目侧包装或 monkeypatch PageIndex 的 `llm_completion()` / `llm_acompletion()` 捕获 usage；如果无法捕获 usage，则禁止在正式答题阶段触发 PageIndex 内部 LLM 调用。

## 5. Implementation Tasks

### Task 1: Project Skeleton And Config

**Files:**

- Create: `agent/__init__.py`
- Create: `agent/config.py`
- Create: `agent/schemas.py`
- Create: `agent/llm_client.py`
- Create: `agent/token_meter.py`
- Create: `scripts/build_preprocess.py`
- Create: `scripts/build_pageindex.py`
- Create: `scripts/run_answers.py`
- Create: `scripts/validate_outputs.py`
- Test: `tests/agent/test_config.py`

- [ ] Step 1: 定义 `AgentConfig`、路径派生函数、PageIndex 构建参数、在线检索预算参数和 CLI 参数。
- [ ] Step 2: 定义基础 usage schema，字段至少包含 `stage/model/prompt_tokens/completion_tokens/total_tokens/latency_ms/success/error`。
- [ ] Step 3: 创建可 mock 的 `LLMClient` 骨架，统一承接后续 Qwen JSON 调用、重试和错误包装。
- [ ] Step 4: 创建 `TokenMeter` 骨架，支持按 `qid/stage` 累加 usage、写 JSONL 日志和生成 `answer.csv` 所需汇总。
- [ ] Step 5: 写测试验证 `--domain insurance --split A` 派生到当前题目文件、PDF 目录、输出目录和日志目录。
- [ ] Step 6: 写测试验证 PageIndex 构建参数和在线检索预算参数存在且默认值符合第 4.1 节。
- [ ] Step 7: 写测试验证 `split` 输出目录统一小写，当前为 `outputs/insurance_a`。
- [ ] Step 8: 运行 `pytest tests/agent/test_config.py -q`，期望通过。
- [ ] Step 9: 提交：

```bash
git add agent scripts tests/agent/test_config.py
git commit -m "feat: add configurable insurance pipeline skeleton"
```

### Task 2: Insurance Domain Profile

**Files:**

- Create: `agent/domain_profiles/__init__.py`
- Create: `agent/domain_profiles/insurance.py`
- Test: `tests/agent/test_domain_profiles.py`

- [ ] Step 1: 定义 `DomainProfile`，包含 `keywords`、`product_aliases`、`liability_terms`、`calculation_patterns`、`quality_thresholds`。
- [ ] Step 2: 在 `insurance.py` 写入当前 16 个文档涉及的产品名和简称，例如平安智盈金生、国寿增益宝、众安白血病医疗险、平安 e 生保、太保团体百万医疗、家财险、特种车险。
- [ ] Step 3: 写测试验证题干中的产品简称能映射到候选产品名。
- [ ] Step 4: 运行 `pytest tests/agent/test_domain_profiles.py -q`，期望通过。
- [ ] Step 5: 提交：

```bash
git add agent/domain_profiles tests/agent/test_domain_profiles.py
git commit -m "feat: add insurance domain profile"
```

### Task 3: PDF Preprocess

**Files:**

- Create: `agent/preprocess.py`
- Modify: `scripts/build_preprocess.py`
- Test: `tests/agent/test_preprocess.py`

- [ ] Step 1: 用 PyMuPDF 扫描 `raw_dir/*.pdf`，按数字文件名排序。
- [ ] Step 2: 为每页写入 `data/processed_data/pages/{domain}/{doc_id}.jsonl`。
- [ ] Step 3: 生成基础 Markdown，标题恢复规则优先识别“第 X 条”“第 X 章”“保险责任”“责任免除”“释义”“现金价值”“退保”“身故保险金”等保险条款信号。
- [ ] Step 4: 写入 `data/processed_data/markdown/{domain}/{doc_id}.page_map.json`，确保每个 Markdown 行能映射到 PDF 物理页。
- [ ] Step 5: 写入 `data/processed_data/quality/{domain}_parse_quality.jsonl`，字段至少包含 `doc_id/page_count/text_pages/char_count/title_count/outline_count/status/error`。
- [ ] Step 6: 写测试验证 16 个 PDF 都有页缓存，当前总页数为 299，且每页 `text` 非空。
- [ ] Step 7: 运行：

```bash
pytest tests/agent/test_preprocess.py -q
python scripts/build_preprocess.py --domain insurance --split A
```

期望：生成 16 个页缓存、16 个 Markdown、16 个 page_map，quality 日志无 fatal error。

- [ ] Step 8: 提交：

```bash
git add agent/preprocess.py scripts/build_preprocess.py tests/agent/test_preprocess.py data/processed_data/pages data/processed_data/markdown data/processed_data/quality
git commit -m "feat: preprocess insurance pdfs into page caches"
```

### Task 4: PageIndex Adapter And Index Build

**Files:**

- Create: `agent/pageindex_adapter.py`
- Modify: `scripts/build_pageindex.py`
- Test: `tests/agent/test_pageindex_adapter.py`

- [ ] Step 1: 在适配器中将 `open_projects/PageIndex` 插入 `sys.path`，直接导入 `pageindex.page_index_md.md_to_tree` 和 `pageindex.page_index.page_index`。
- [ ] Step 2: 实现 Markdown 主路，显式关闭 `node_summary/doc_description/node_text`，保留 `node_id`。
- [ ] Step 3: 实现 PDF 兜底路，显式传入模型、目录页检查页数、单节点最大页数和开关。
- [ ] Step 4: 写测试 monkeypatch `md_to_tree()` 和 `page_index()`，断言调用参数完全匹配第 4.2 节契约。
- [ ] Step 5: 写测试 monkeypatch `PageIndexClient.index()`，如果被调用则测试失败。
- [ ] Step 6: 运行 `pytest tests/agent/test_pageindex_adapter.py -q`，期望通过。
- [ ] Step 7: 提交：

```bash
git add agent/pageindex_adapter.py scripts/build_pageindex.py tests/agent/test_pageindex_adapter.py
git commit -m "feat: wrap pageindex with explicit competition settings"
```

### Task 5: Node Spans And Index Quality

**Files:**

- Create: `agent/index_store.py`
- Modify: `agent/pageindex_adapter.py`
- Modify: `scripts/build_pageindex.py`
- Test: `tests/agent/test_index_store.py`

- [ ] Step 1: 从 Markdown 主索引读取 `node_id/title/line_num/nodes`。
- [ ] Step 2: 用“当前标题行到下一个同级或更高层级标题前一行”的规则生成 `start_line/end_line`。
- [ ] Step 3: 通过 `page_map.line_to_page` 转换为 `start_page/end_page/source_page_range`。
- [ ] Step 4: 写入 `data/processed_data/pageindex/{domain}/{doc_id}.node_spans.json`。
- [ ] Step 5: 质量检查至少输出 `node_count/empty_title_count/bad_page_range_count/keyword_title_hits/page_mapping_coverage/index_source/status`。
- [ ] Step 6: 当 Markdown 主路质量不合格时，触发 PDF 兜底或写入 `page_keyword` 降级状态。
- [ ] Step 7: 写测试覆盖 `line_num` 不暴露给 `get_document_structure()`，坏页码范围被标记为不可用。
- [ ] Step 8: 运行：

```bash
pytest tests/agent/test_index_store.py -q
python scripts/build_pageindex.py --domain insurance --split A
```

期望：16 个文档都有主索引尝试记录；可用主索引有 node_spans；不合格文档有兜底或降级记录。

- [ ] Step 9: 提交：

```bash
git add agent/index_store.py agent/pageindex_adapter.py scripts/build_pageindex.py tests/agent/test_index_store.py data/processed_data/pageindex data/processed_data/quality
git commit -m "feat: build pageindex node spans and quality reports"
```

### Task 6: Document Catalog

**Files:**

- Create: `agent/catalog.py`
- Create: `scripts/build_catalog.py`
- Test: `tests/agent/test_catalog.py`

- [ ] Step 1: 从 PDF 文件名、Markdown 标题、题目中出现的产品名和保险公司信号生成 `doc_catalog.jsonl`。
- [ ] Step 2: 每行至少包含 `doc_id/product_name/aliases/insurer/insurance_type/source_pdf/top_titles/primary_index_route`。
- [ ] Step 3: A 榜要求 catalog 覆盖 `1` 到 `16`，且题目中的 `doc_ids` 全部可查。
- [ ] Step 4: 写测试验证 catalog 字段完整、`doc_id` 唯一、覆盖题目中出现的全部 `doc_ids`。
- [ ] Step 5: 运行：

```bash
pytest tests/agent/test_catalog.py -q
python scripts/build_catalog.py --domain insurance --split A
```

期望：`data/processed_data/catalog/doc_catalog.jsonl` 有 16 行，全部 `doc_id` 唯一。

- [ ] Step 6: 提交：

```bash
git add agent/catalog.py scripts/build_catalog.py tests/agent/test_catalog.py data/processed_data/catalog
git commit -m "feat: build insurance document catalog"
```

### Task 7: Question Parsing And A-Split Doc Retrieval

**Files:**

- Create: `agent/question_parser.py`
- Create: `agent/doc_retriever.py`
- Test: `tests/agent/test_question_parser.py`
- Test: `tests/agent/test_doc_retriever.py`

- [ ] Step 1: 解析 `qid/domain/split/question/options/answer_format/type/doc_ids`。
- [ ] Step 2: 从题干和选项抽取产品名、责任类别、金额、比例、年度、免赔额、账户价值、现金价值等信号。
- [ ] Step 3: 结合 catalog 输出 `mentioned_products` 和 `doc_product_map`。
- [ ] Step 4: `DocRetriever.retrieve()` 对 `split=A` 固定返回题目内 `doc_ids`，保留原顺序，写 warning 但不补充题外文档。
- [ ] Step 5: 运行：

```bash
pytest -q tests/agent/test_question_parser.py tests/agent/test_doc_retriever.py
```

期望：20 道保险题均能解析；所有返回文档 ID 均来自原题。

- [ ] Step 6: 提交：

```bash
git add agent/question_parser.py agent/doc_retriever.py tests/agent/test_question_parser.py tests/agent/test_doc_retriever.py
git commit -m "feat: parse insurance questions and preserve A split doc ids"
```

### Task 8: Tree Retrieval

**Files:**

- Create: `agent/tree_retriever.py`
- Modify: `agent/index_store.py`
- Test: `tests/agent/test_index_store.py`

- [ ] Step 1: `IndexStore.get_document_metadata()` 返回 `doc_id/product_name/index_status/page_count/index_source`。
- [ ] Step 2: `IndexStore.get_document_structure()` 返回去掉 `text` 的紧凑树，并统一 `page_range`。
- [ ] Step 3: `TreeRetriever` 先用规则预筛命中产品名、责任类别、标题关键词、金额/比例词的节点。
- [ ] Step 4: 小树直接交给 Qwen 选择节点；大树只把预筛节点交给 Qwen。
- [ ] Step 5: 输出 `CandidateNode`，每个文档最多 `MAX_NODES_PER_DOC=5`，每文档页数最多 `MAX_PAGES_PER_DOC=8`。
- [ ] Step 6: 写测试验证 `CandidateNode.needs_page_fetch` 恒为 `true`，且 `reason` 不被当成证据结论。
- [ ] Step 7: 运行 `pytest tests/agent/test_index_store.py -q`，期望通过。
- [ ] Step 8: 提交：

```bash
git add agent/tree_retriever.py agent/index_store.py tests/agent/test_index_store.py
git commit -m "feat: retrieve candidate nodes from compact pageindex trees"
```

### Task 9: Evidence Extraction

**Files:**

- Create: `agent/evidence_extractor.py`
- Extend: `agent/llm_client.py`
- Extend: `agent/token_meter.py`
- Test: `tests/agent/test_validate_outputs.py`

- [ ] Step 1: 对每个 `CandidateNode` 调用 `IndexStore.get_page_content(doc_id, page_range)` 读取页级原文。
- [ ] Step 2: 对每个选项分别抽取证据 verdict，取值只能是 `support/refute/unclear`。
- [ ] Step 3: 去重相同 `doc_id + pages + quote` 的证据。
- [ ] Step 4: 每个选项至少保留 1 条 verdict；被最终选择的选项必须能关联 `support` 证据。
- [ ] Step 5: 写入 evidence 时保留 `quote`、`normalized_fact`、`numbers`、`confidence`。
- [ ] Step 6: 写测试验证证据记录能追溯到 `doc_id + node_id/pages + quote`。
- [ ] Step 7: 运行 `pytest tests/agent/test_validate_outputs.py -q`，期望通过。
- [ ] Step 8: 提交：

```bash
git add agent/evidence_extractor.py agent/llm_client.py agent/token_meter.py tests/agent/test_validate_outputs.py
git commit -m "feat: extract option-level evidence from page text"
```

### Task 10: Calculation Engine

**Files:**

- Create: `agent/calculation.py`
- Test: `tests/agent/test_calculation.py`

- [ ] Step 1: 实现金额单位归一化，支持元、万元、百分比、保单年度。
- [ ] Step 2: 覆盖当前保险题高频计算：身故保险金比较、退保所得、医疗费用扣除医保和免赔额、赔付比例与最高限额、排序比较。
- [ ] Step 3: 计算结果记录 `inputs/formula/result/unit/source_evidence_ids`。
- [ ] Step 4: 写测试覆盖 `ins_a_001` 风格排序和 `ins_a_003` 风格医疗费用扣减。
- [ ] Step 5: 运行 `pytest tests/agent/test_calculation.py -q`，期望通过。
- [ ] Step 6: 提交：

```bash
git add agent/calculation.py tests/agent/test_calculation.py
git commit -m "feat: add deterministic insurance calculation engine"
```

### Task 11: Answer Judging And Output Validation

**Files:**

- Create: `agent/answer_judge.py`
- Modify: `scripts/validate_outputs.py`
- Test: `tests/agent/test_answer_judge.py`
- Test: `tests/agent/test_validate_outputs.py`

- [ ] Step 1: `AnswerJudge` 只基于证据记忆和计算结果判断，不直接使用树摘要。
- [ ] Step 2: `mcq` 输出唯一大写字母；`multi` 输出去重排序大写字母串；`tf` 输出 A 或 B。
- [ ] Step 3: 非法答案触发程序化修正；修正失败时写 `unclear`、warning 和补检索状态。
- [ ] Step 4: `validate_outputs.py` 校验 `answer.csv` 第一行为 `summary`，后续 20 行覆盖题库 `qid`。
- [ ] Step 5: `validate_outputs.py` 校验每个最终选中选项至少有一条 `support` 证据。
- [ ] Step 6: 运行：

```bash
pytest tests/agent/test_answer_judge.py tests/agent/test_validate_outputs.py -q
```

期望：答案格式校验、summary usage 校验、证据追溯校验全部通过。

- [ ] Step 7: 提交：

```bash
git add agent/answer_judge.py scripts/validate_outputs.py tests/agent/test_answer_judge.py tests/agent/test_validate_outputs.py
git commit -m "feat: judge and validate formatted insurance answers"
```

### Task 12: End-To-End Pipeline

**Files:**

- Create: `agent/pipeline.py`
- Modify: `scripts/run_answers.py`
- Extend: `agent/token_meter.py`
- Test: `tests/agent/test_validate_outputs.py`

- [ ] Step 1: `pipeline.py` 编排单题：parse -> doc retrieve -> structure -> tree retrieve -> evidence -> calculation -> answer -> audit。
- [ ] Step 2: 全量运行读取 20 道题，按题目文件顺序写出 `answer.csv`。
- [ ] Step 3: `TokenMeter` 汇总所有 Qwen 调用，写题目行 usage 和 `summary` usage。
- [ ] Step 4: 回退策略实现：节点不足扩页一次、证据不足补检索一次、答案非法自检一次，均记录到 evidence/logs。
- [ ] Step 5: 运行：

```bash
python scripts/run_answers.py --domain insurance --split A
python scripts/validate_outputs.py --domain insurance --split A
```

期望：`outputs/insurance_a/answer.csv`、`outputs/insurance_a/evidence.jsonl` 和日志生成，验证脚本退出码为 0。

- [ ] Step 6: 提交：

```bash
git add agent/pipeline.py scripts/run_answers.py agent/token_meter.py outputs/insurance_a
git commit -m "feat: run end-to-end insurance A split pipeline"
```

### Task 13: Reproducibility Documentation

**Files:**

- Create or modify: `README.md`
- Modify: `requirements.txt`
- Modify: `docs/pageindex_finance_gpt5.5pro/A榜保险PageIndex系统PLAN.md`

- [ ] Step 1: 写明安装依赖、环境变量、正式 Qwen 模型配置和开发 ARK 配置边界。
- [ ] Step 2: 写明一键复现命令：

```bash
python scripts/build_preprocess.py --domain insurance --split A
python scripts/build_pageindex.py --domain insurance --split A
python scripts/build_catalog.py --domain insurance --split A
python scripts/run_answers.py --domain insurance --split A
python scripts/validate_outputs.py --domain insurance --split A
```

- [ ] Step 3: 写明不会读取 `.tmp_del`、不会修改 PageIndex、不会把非 Qwen 推理结果用于正式答题。
- [ ] Step 4: 运行完整测试：

```bash
pytest tests/agent -q
```

期望：全部通过。

- [ ] Step 5: 提交：

```bash
git add README.md requirements.txt docs/pageindex_finance_gpt5.5pro/A榜保险PageIndex系统PLAN.md
git commit -m "docs: document reproducible insurance pageindex workflow"
```

## 6. Testing Matrix

| 测试 | 命令 | 验收点 |
| --- | --- | --- |
| 配置 | `pytest tests/agent/test_config.py -q` | domain/split/path/model 派生正确。 |
| 领域 profile | `pytest tests/agent/test_domain_profiles.py -q` | 保险关键词、产品别名、责任类别和质量阈值可用。 |
| 预处理 | `pytest tests/agent/test_preprocess.py -q` | 16 个 PDF、299 页、页文本非空、page_map 完整。 |
| PageIndex 适配 | `pytest tests/agent/test_pageindex_adapter.py -q` | 只调用底层函数，显式关闭摘要/正文/描述。 |
| 索引存储 | `pytest tests/agent/test_index_store.py -q` | `line_num` 不外泄；`page_range` 合法；坏节点标记。 |
| catalog | `pytest tests/agent/test_catalog.py -q` | catalog 覆盖 16 个 doc_id，字段完整且 doc_id 唯一。 |
| 题目解析 | `pytest -q tests/agent/test_question_parser.py tests/agent/test_doc_retriever.py` | 产品别名、数值条件、选项关键词可解析。 |
| 文档检索 | `pytest -q tests/agent/test_question_parser.py tests/agent/test_doc_retriever.py` | A 榜只返回题目 `doc_ids`。 |
| 计算引擎 | `pytest tests/agent/test_calculation.py -q` | 金额、比例、扣减和排序计算可复现。 |
| 答案合法化 | `pytest -q tests/agent/test_answer_judge.py tests/agent/test_validate_outputs.py` | `mcq/multi/tf` 输出合法。 |
| 输出校验 | `pytest tests/agent/test_validate_outputs.py -q` | `answer.csv`、`evidence.jsonl` schema 与追溯合格。 |
| 全量单元测试 | `pytest tests/agent -q` | 全部通过。 |
| 预处理运行 | `python scripts/build_preprocess.py --domain insurance --split A` | 生成 pages/markdown/page_map/parse quality。 |
| 索引运行 | `python scripts/build_pageindex.py --domain insurance --split A` | 生成 PageIndex 树、node_spans、index quality。 |
| catalog 运行 | `python scripts/build_catalog.py --domain insurance --split A` | catalog 覆盖 16 个 doc_id。 |
| 答题运行 | `python scripts/run_answers.py --domain insurance --split A` | 生成 answer/evidence/logs。 |
| 最终验收 | `python scripts/validate_outputs.py --domain insurance --split A` | 退出码 0。 |

## 7. Acceptance Criteria

数据与索引：

- 16 个保险 PDF 均生成页级缓存。
- 页缓存总页数为 299，所有页有可抽取文本。
- 每个文档均尝试 Markdown 主索引；主索引不合格时记录原因并触发 PDF 兜底或页级关键词降级。
- 可用 Markdown 主索引都有 `node_spans`，所有暴露给在线模块的节点都有 PDF `page_range`。
- 默认主索引关闭节点摘要；紧凑树中的 `summary/prefix_summary` 仅为 optional/nullable 字段，不作为默认检索依据。
- `doc_catalog.jsonl` 覆盖 16 个 `doc_id`，且能校验 A 榜题目中的 `doc_ids`。

问答链路：

- 20 道 `ins_a_*` 题完整走通。
- 每题候选文档严格等于题目内 `doc_ids` 的子集或原集，不引入题外文档。
- 树检索只用于定位候选页段，不直接定性选项。
- 证据判断必须基于页级原文。
- 每个选项都有 `support/refute/unclear` verdict。
- 每个最终选中选项至少有一条 `support` 证据。
- 计算题和排序题保留中间事实、单位归一化、公式和计算结果。

输出与审计：

- `answer.csv` 第一行为 `summary`，其余 20 行覆盖全部题目。
- `mcq` 答案为一个大写字母。
- `multi` 答案为去重排序大写字母串，无空格、逗号或重复字母。
- 每题 `prompt_tokens/completion_tokens/total_tokens` 为非负整数。
- `summary.total_tokens` 等于题目行 Token 之和。
- `evidence.jsonl` 每题一行，包含 `qid/answer/candidate_docs/selected_nodes/evidence/calculations/usage/fallbacks/warnings/option_judgements`。
- 日志包含 `qid/stage/model/prompt_tokens/completion_tokens/total_tokens/latency_ms/success/error`。
- PageIndex 离线构建日志与在线答题 Token 分开记录；若正式答题阶段触发 PageIndex 内部 LLM 调用，必须计入对应题目 usage。
- 日志不保存 API Key、标准答案或题外答案信息。

合规与扩展：

- 不读取 `.tmp_del`。
- 不修改 `open_projects/PageIndex`。
- 正式推理问答阶段只使用 Qwen 系列模型。
- 开发 ARK 调试结果不进入正式索引、检索、证据或答案产物。
- 项目侧适配器必须显式覆盖 PageIndex 默认模型、摘要、正文、文档描述和 PDF 构树预算参数。
- 保险专属逻辑集中在 `agent/domain_profiles/insurance.py`、catalog 和计算规则中。
- B 榜扩展点仅体现在 `DocRetriever` 接口和 catalog，不影响 A 榜主流程。

## 8. Risk Controls

| 风险 | 处理 |
| --- | --- |
| PDF outline 缺失 | Markdown 主路基于保险标题恢复结构；质量不足才走 PDF 兜底。 |
| Markdown 行号被误当页码 | `IndexStore` 只暴露 `page_range`，测试禁止 `line_num` 外泄。 |
| PageIndex 默认模型/开关不合规 | 适配器显式覆盖赛题相关参数；测试禁止高层 `PageIndexClient.index()`，并断言不依赖默认模型和摘要开关。 |
| 关闭摘要后树检索信号不足 | 默认使用 `title/page_range/层级/保险关键词/题干和选项信号`；不得把空 `summary` 当作有效依据。 |
| 非 Qwen 调试污染正式结果 | 配置区分 `inference_model` 与 `dev_model`；正式命令默认 Qwen。 |
| PageIndex 内部 LLM usage 缺失 | 离线构建单独记录；在线触发时必须包装或 monkeypatch PageIndex LLM 函数捕获 usage，否则禁止在线触发。 |
| 多选题漏选/多选 | `AnswerJudge` 对每个选项单独 verdict，最后程序化去重排序。 |
| 计算题模型口算错误 | `CalculationEngine` 统一处理金额、比例、扣减和排序。 |
| Token 统计缺失 | 所有 LLM 调用必须通过 `LLMClient` 和 `TokenMeter`。 |
| 证据不可追溯 | evidence 强制保留 `doc_id + node_id/pages + quote`。 |

## 9. Execution Order

推荐按以下顺序执行并验证，每个任务完成后提交一次：

1. Task 1：项目骨架与配置。
2. Task 2：保险领域 profile。
3. Task 3：PDF 预处理。
4. Task 4：PageIndex 适配器。
5. Task 5：node_spans 与质量报告。
6. Task 6：doc catalog。
7. Task 7：题目解析与 A 榜文档 passthrough。
8. Task 8：树检索。
9. Task 9：证据抽取。
10. Task 10：计算引擎。
11. Task 11：答案判断与输出校验。
12. Task 12：端到端流水线。
13. Task 13：复现文档。

最终验收命令：

```bash
pytest tests/agent -q
python scripts/build_preprocess.py --domain insurance --split A
python scripts/build_pageindex.py --domain insurance --split A
python scripts/build_catalog.py --domain insurance --split A
python scripts/run_answers.py --domain insurance --split A
python scripts/validate_outputs.py --domain insurance --split A
```

完成标准：上述命令全部退出码为 0，且 `outputs/insurance_a/answer.csv`、`outputs/insurance_a/evidence.jsonl`、`outputs/insurance_a/logs/` 存在并满足第 7 节验收标准。
