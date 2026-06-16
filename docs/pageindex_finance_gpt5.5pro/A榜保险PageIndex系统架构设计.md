# A榜保险 PageIndex 系统架构设计

## 1. 设计目标

本文档定义一个面向 A 榜保险条款题目的 PageIndex 问答系统架构。当前任务只处理：

- 题目范围：`data/public_dataset_upload/questions/group_a/insurance_questions.json`
- 文档范围：`data/public_dataset_upload/raw/insurance` 下的 16 个 PDF
- 领域范围：`domain=insurance`
- 榜单范围：`split=A`

系统目标是在不修改 `open_projects/PageIndex` 源码的前提下，把 PageIndex 作为文档内部结构索引组件，用于定位保险条款证据页段；项目侧负责题目解析、节点选择、证据抽取、计算、答案判断、Token 统计和结果导出。

当前版本不实现 B 榜候选文档召回，但在配置、数据产物和模块边界上保留扩展点，后续可扩展到 B 榜以及监管法规、金融合同、财务报表、行业研报等领域。

## 2. 核心约束

| 约束                        | 设计处理                                                                       |
| ------------------------- | -------------------------------------------------------------------------- |
| A 榜题目提供 `doc_ids`         | 主流程直接使用题目给定文档，不做候选文档召回。                                                    |
| 保险数据为 16 个 PDF            | 离线阶段为每个 PDF 生成页级缓存和 PageIndex 树索引。                                         |
| 多数保险 PDF 没有可靠 PDF 目录      | 采用 Markdown 优先、PDF 直连兜底的双路线并行索引策略。                                         |
| PageIndex 是第三方源码          | 只通过薄封装调用，不修改 `open_projects/PageIndex`。                                    |
| 正式推理问答阶段模型限制              | 文档定位、证据判断、答案生成、自检等正式推理调用使用 Qwen 系列并统计 Token。                               |
| `docs/模型配置.md` 中存在 ARK 配置 | ARK 兼容 OpenAI 协议配置仅作为开发调试配置，不作为正式赛题推理模型假设。                                 |
| PageIndex 默认配置不是赛题配置      | 不使用默认 `config.yaml` 和 `PageIndexClient.index()` 作为正式索引入口，项目侧必须显式传入模型与开关参数。 |

## 3. 总体架构

系统分为六层：

```text
原始保险 PDF
  -> 文档解析层
  -> PageIndex 结构索引层
  -> 保险题检索层
  -> 证据记忆层
  -> 答案推理层
  -> 输出与审计层
```

### 3.1 文档解析层

职责：

- 扫描 `data/public_dataset_upload/raw/insurance/*.pdf`。
- 为每个 `doc_id` 生成页级文本缓存。
- 尝试恢复保险条款章节结构，生成带页码映射的 Markdown。
- 记录解析质量，例如页数、字符数、标题命中数、目录恢复方式、失败原因。

产物：

- `data/processed_data/pages/{domain}/{doc_id}.jsonl`
- `data/processed_data/markdown/insurance/{doc_id}.md`
- `data/processed_data/markdown/insurance/{doc_id}.page_map.json`
- `data/processed_data/quality/insurance_parse_quality.jsonl`

### 3.2 PageIndex 结构索引层

职责：

- 使用 `open_projects/PageIndex/pageindex/page_index_md.py` 的 Markdown 树能力生成主索引。
- 使用 `open_projects/PageIndex/pageindex/page_index.py` 的 PDF 树能力生成兜底索引。
- 为每个索引文件保留 `doc_id`、源文件路径、索引路线、树结构和页码映射信息。
- 输出统一格式，屏蔽 PDF 路线与 Markdown 路线的结构差异。

调用边界：

- 禁止在正式流程中直接使用 `PageIndexClient.index()`，因为该高层入口会硬编码开启节点摘要、文档描述和节点正文，且会生成随机 workspace 文档 ID，不符合赛题 `doc_id` 追踪要求。
- Markdown 主路必须由项目侧薄封装直接调用 `md_to_tree()`，并显式设置 `if_add_node_summary="no"`、`if_add_doc_description="no"`、`if_add_node_text="no"`、`if_add_node_id="yes"`。
- PDF 兜底路必须由项目侧薄封装直接调用 `page_index()`，并显式设置 `if_add_node_summary="no"`、`if_add_doc_description="no"`、`if_add_node_text="no"`、`if_add_node_id="yes"`，正式提交产物必须传入赛题允许的 Qwen 模型。
- 不依赖 `open_projects/PageIndex/pageindex/config.yaml` 默认值；该文件中的默认模型和摘要开关只视为 PageIndex 项目自身示例配置。

产物：

- `data/processed_data/pageindex/insurance/{doc_id}.json`
- `data/processed_data/pageindex/insurance/{doc_id}.pdf_fallback.json`
- `data/processed_data/pageindex/insurance/{doc_id}.node_spans.json`
- `data/processed_data/quality/insurance_index_quality.jsonl`

### 3.3 保险题检索层

职责：

- 读取 A 榜题目中的 `doc_ids`。
- 按题干和选项生成保险领域查询信号。
- 在每个指定文档的 PageIndex 树中选择少量候选节点。
- 将候选节点转换为页码窗口，读取紧凑原文页段。

当前 A 榜不启用候选文档召回。`DocRetriever` 接口保留为空实现或 passthrough 实现：输入题目，输出题目内的 `doc_ids`。

### 3.4 证据记忆层

职责：

- 按选项抽取证据，而不是只按题目抽取证据。
- 将页段原文压缩为可审计的结构化证据记忆。
- 去重同一 `doc_id + pages + quote` 的重复证据。
- 保留支持、反驳、不确定三类证据，避免过早丢弃冲突信息。

证据记忆必须保留原文短引文、页码、节点和归一化事实。压缩只压缩解释，不删除定位信息和数字字段。

### 3.5 答案推理层

职责：

- 基于证据记忆逐项判断 A/B/C/D。
- 对金额、比例、排序题调用程序化计算模块，不依赖模型口算。
- 对单选、多选、判断题做答案合法化。
- 在证据不足、答案为空、选项冲突时触发一次补检索或自检。

### 3.6 输出与审计层

职责：

- 导出 `answer.csv`。
- 导出 `evidence.jsonl`。
- 记录模型调用日志、Token usage、阶段耗时和异常信息。
- 输出可复现运行摘要，便于赛后代码审核。

## 4. 双路线索引设计

### 4.1 Markdown 优先路线

Markdown 路线是默认主路：

```text
PDF -> 页级文本/OCR/版面恢复 -> 带标题层级的 Markdown -> PageIndex Markdown tree -> node_id -> line_num -> source_page
```

适用原因：

- 保险条款常有“第几章”“第几条”“保险责任”“责任免除”“释义”等稳定标题。
- 当前 insurance 统计显示多数 PDF 内置目录不可用，直接依赖 PDF outline 风险高。
- Markdown 标题层级可控，便于修正条款编号、跨页标题和表格标题。
- 不生成摘要时，Markdown 树构建成本更可控。

关键要求：

- Markdown 标题必须能映射回 PDF 页码。
- `line_num` 不能直接当作页码使用，必须通过 `page_map` 转换。
- Markdown 路线的索引文件应额外保存 `node_id -> source_page_range`。
- 不直接使用 PageIndex `retrieve.py` 的 Markdown `get_page_content()` 作为正式页读取工具；该函数把 `pages` 解释为 Markdown 行号，不是 PDF 物理页码。
- 正式答题统一从项目侧 `data/processed_data/pages/{domain}/{doc_id}.jsonl` 读取 PDF 页文本。

Markdown 主路必须生成节点跨度文件 `node_spans`。字段约定：

| 字段                  | 说明                       |
| ------------------- | ------------------------ |
| `doc_id`            | 赛题文档 ID，例如 `1`、`16`      |
| `node_id`           | PageIndex 节点 ID          |
| `title`             | 节点标题                     |
| `start_line`        | 节点标题所在 Markdown 行        |
| `end_line`          | 下一个同级或更高层级标题前一行；末节点为文档末行 |
| `start_page`        | `start_line` 映射到的 PDF 页  |
| `end_page`          | `end_line` 映射到的 PDF 页    |
| `source_page_range` | 面向后续模块的页码范围字符串，例如 `4-6`  |

`InsuranceDocStore` 只能向后续模块暴露 `source_page_range`，不能让后续问答模块直接消费 `line_num`。

### 4.2 PDF 直连兜底路线

PDF 路线作为快速验证和兜底：

```text
PDF -> PageIndex PDF tree -> node_id -> start_index/end_index -> PDF page text
```

适用场景：

- Markdown 解析失败。
- Markdown 标题恢复质量过低。
- PDF 文本抽取已足够稳定，且 PageIndex 能生成合理章节树。
- 需要快速构建最小可用基线。

限制：

- 无目录保险条款可能生成较粗的节点。
- `start_index/end_index` 可能受页码偏移和标题定位影响。
- PDF 路线如在正式流程中调用 LLM 构树，应使用 Qwen 并记录构建日志；更推荐把索引构建放到题目无关的离线阶段。
- PDF 路线即使关闭摘要，也可能调用 LLM 进行目录检测、目录转换、无目录树生成和页码校验，不能把它视为零成本兜底。

### 4.3 路线选择规则

每个文档离线构建后做质量评分：

| 指标      | 推荐判断                                        |
| ------- | ------------------------------------------- |
| 节点数     | 太少说明章节恢复不足，优先检查 Markdown 标题或 PDF 兜底。        |
| 空标题数    | 非 0 时记录质量问题。                                |
| 页码范围    | `start > end` 或超出页数时判为坏节点。                  |
| 关键词覆盖   | “保险责任”“责任免除”“现金价值”“犹豫期”等标题命中越多越优先。          |
| 页码映射完整度 | Markdown 节点缺少 `source_page_range` 时不得作为主索引。 |

默认优先级：

1. Markdown 主索引质量合格时使用 Markdown。
2. Markdown 不合格但 PDF 索引合格时使用 PDF。
3. 两者均不合格时降级为页级关键词检索，并把该文档标记为需人工检查或强化解析。

PDF 兜底索引默认采用按质量触发策略：先构建 Markdown 主索引并运行质量检查，仅对 Markdown 缺少有效页码映射、关键标题覆盖不足或坏节点过多的文档构建 PDF 兜底。若为了实验选择 16 个 PDF 全量构建兜底索引，必须单独写入索引构建日志，并明确该成本不混入每题正式答题 Token。

## 5. 数据与配置入口

### 5.1 固定配置

| 配置项              | 值                                                                       |
| ---------------- | ----------------------------------------------------------------------- |
| `domain`         | `insurance`                                                             |
| `split`          | `A`                                                                     |
| `raw_dir`        | `data/public_dataset_upload/raw/insurance`                              |
| `questions`      | `data/public_dataset_upload/questions/group_a/insurance_questions.json` |
| `pageindex_root` | `open_projects/PageIndex`                                               |
| `output_dir`     | `outputs/insurance_a`                                                   |

### 5.2 目录布局

```text
data/processed_data/
├── pages/
│   └── {doc_id}.jsonl
├── markdown/
│   └── insurance/
│       ├── {doc_id}.md
│       └── {doc_id}.page_map.json
├── pageindex/
│   └── insurance/
│       ├── {doc_id}.json
│       ├── {doc_id}.pdf_fallback.json
│       └── {doc_id}.node_spans.json
├── catalog/
│   └── doc_catalog.jsonl
└── quality/
    ├── insurance_parse_quality.jsonl
    └── insurance_index_quality.jsonl

outputs/
└── insurance_a/
    ├── answer.csv
    ├── evidence.jsonl
    └── logs/
```

`doc_catalog.jsonl` 是 A 榜也必须使用的元数据输入。A 榜虽然不需要候选文档召回，但题目和选项使用产品名、险种简称和别名表达，题目字段只给数字 `doc_ids`；没有 catalog，选项级检索容易把产品和文档对应关系搞错。

保险 catalog 每行至少包含：

| 字段                    | 说明                                         |
| --------------------- | ------------------------------------------ |
| `doc_id`              | 数字文档 ID，与 PDF 文件名一致                        |
| `product_name`        | 条款主产品名                                     |
| `aliases`             | 题干、选项中可能出现的简称或别名                           |
| `insurer`             | 保险公司或承保主体                                  |
| `insurance_type`      | 年金险、医疗险、重疾险、责任险、家财险、车险等                    |
| `source_pdf`          | 原始 PDF 相对路径                                |
| `top_titles`          | 一级/二级章节标题，来自解析或 PageIndex 树                |
| `primary_index_route` | `markdown`、`pdf_fallback` 或 `page_keyword` |

## 6. Agent 模块边界

A 榜保险 Agent 的核心不是传统向量检索，而是 PageIndex 风格的结构索引推理式检索：LLM 先消费 PageIndex 树中的 `title + summary + page_range`，基于题干、选项和保险领域信号定位候选节点；只有在节点定位完成后，才按紧凑页码范围读取页级原文并抽取证据。树摘要只用于定位证据页，不用于直接判断选项真假或生成最终答案。

### 6.1 `QuestionParser`

输入：题目 JSON。

输出：

- `qid`
- `question`
- `options`
- `answer_format`
- `type`
- `domain`
- `doc_ids`
- 题干关键词、选项关键词、数字条件、产品名候选

保险题应特别识别：

- 产品名和别名：平安智盈金生、国寿增益宝、国寿鑫享添盈、平安富鸿金生、众安白血病医疗险、平安 e 生保、太保团体百万医疗、平安安佑福重疾险、平安预防接种意外险、众安食责险、众安营运交通意外险、平安特种车险、众安特种车险、平安家财险、众安家财险等。
- 责任类别：身故、退保、医疗费用、等待期、免赔额、保单贷款等。
- 计算条件：已交保费、现金价值、账户价值、免赔额、报销金额、给付比例。

`QuestionParser` 必须结合 `doc_catalog.jsonl` 输出 `mentioned_products` 和 `doc_product_map`，把题目中的产品名、简称、别名映射到题目给定的 `doc_ids`。

### 6.2 `DocRetriever`

当前 A 榜行为：

- 直接返回题目内 `doc_ids`。
- 不做跨文档召回、不做重排。
- 用 `doc_catalog.jsonl` 校验题目涉及的产品名是否能映射到这些 `doc_ids`；无法映射时写入 warning，但不得自行引入题目外文档。

预留 B 榜行为：

- 按 `domain` 过滤 `doc_catalog`。
- 基于产品名、公司名、法规名、年份、标题关键词做候选召回。
- 使用 Qwen 对紧凑 catalog 做候选文档选择。

### 6.3 `InsuranceDocStore`

职责：

- 根据 `doc_id` 读取 PageIndex 主索引；主索引不可用时读取 PDF 兜底索引。
- 提供 PageIndex 风格的三个只读接口：先读文档元数据，再读去正文的紧凑树结构，最后按页码范围读取原文。
- 屏蔽 Markdown `line_num` 与 PDF `start_index/end_index` 的差异，后续模块只消费 PDF 页码范围。

统一输出应使用页码范围，不把行号暴露给后续问答模块。`InsuranceDocStore` 只提供页级原文读取接口；在线答题链路中只有 `EvidenceExtractor` 调用该接口并基于原文定性证据。

公开接口：

| 接口                                | 输出                                                                                | 说明                      |
| --------------------------------- | --------------------------------------------------------------------------------- | ----------------------- |
| `get_document_metadata(doc_id)`   | `{doc_id, product_name, doc_description, index_status, page_count, index_source}` | 确认文档身份、索引状态、页数和索引来源。    |
| `get_document_structure(doc_id)`  | compact PageIndex tree without raw text                                           | 返回去掉正文 `text` 字段的紧凑树结构。 |
| `get_page_content(doc_id, pages)` | page text list                                                                    | 只按紧页码范围读取页级原文。          |

`get_document_structure` 返回的紧凑树结构只保留：

- `node_id`
- `title`
- `summary`
- `page_range`
- 层级关系
- `index_source`

不得在紧凑树中携带节点正文 `text`。如果底层是 Markdown 主索引，`InsuranceDocStore` 必须通过 `node_spans` 把 `line_num` 转换为 `source_page_range`；如果底层是 PDF 兜底索引，则从节点 `start_index/end_index` 归一化为同一套 `page_range` 字段。

Markdown 主索引读取规则：

- 从 `{doc_id}.json` 读取 PageIndex 树。
- 从 `{doc_id}.node_spans.json` 读取 `node_id -> source_page_range`。
- 从 `data/processed_data/pages/{domain}/{doc_id}.jsonl` 读取 PDF 页文本。

PDF 兜底索引读取规则：

- 从 `{doc_id}.pdf_fallback.json` 读取 PageIndex 树。
- 使用节点 `start_index/end_index` 生成页码范围。
- 页文本仍优先读取 `data/processed_data/pages/{domain}/{doc_id}.jsonl`，只在缓存缺失时才回读原始 PDF。

### 6.4 `TreeRetriever`

职责：

- 输入 `QuestionParser` 结果、候选 `doc_ids`、`InsuranceDocStore.get_document_structure` 返回的紧凑树结构。
- 基于结构索引做节点选择，不读取页级正文，不直接回答题目。
- 输出可供后续取页的候选节点列表。

小树策略：

- 让 Qwen 直接阅读完整紧凑树结构。
- 依据题干、选项、产品名、责任类别、数字条件选择节点。
- 只输出需要取原文验证的节点，不判断选项真假。

大树策略：

- 先用规则预筛保留命中产品名/别名、责任标题、保险关键词、选项关键词、金额比例词的节点。
- 再让 Qwen 在预筛候选节点内做结构索引推理式选择。
- 预筛和模型选择都只能使用 `title`、`summary`、`page_range` 和层级关系。

公开接口：

`TreeRetriever.retrieve(parsed_question, doc_id, compact_tree) -> list[CandidateNode]`

`CandidateNode` 至少包含：

| 字段                 | 说明                       |
| ------------------ | ------------------------ |
| `doc_id`           | 文档 ID                    |
| `node_id`          | PageIndex 节点 ID          |
| `title`            | 节点标题                     |
| `page_range`       | PDF 页码范围                 |
| `matched_signals`  | 命中的产品名、责任类别、关键词、数字条件等    |
| `reason`           | 为什么该节点可能含有证据             |
| `needs_page_fetch` | 固定为 `true`，表示必须读取原文后才能定性 |

约束：

- 不得输出最终答案。
- 不得基于树摘要判断选项真假。
- 不得把节点选择理由当作证据结论。
- 树摘要只用于定位证据页。

### 6.5 `EvidenceExtractor`

职责：

- 接收 `TreeRetriever` 输出的 `CandidateNode`，调用 `InsuranceDocStore.get_page_content(doc_id, pages)` 读取页级原文。
- 对每个选项分别基于原文页段生成证据判断。
- 只基于页段原文判断证据是支持、反驳还是不确定，不能基于 `TreeRetriever` 的选择理由直接定性。
- 输出结构化证据记忆。

证据字段：

| 字段                | 说明                           |
| ----------------- | ---------------------------- |
| `qid`             | 题目 ID                        |
| `doc_id`          | 文档 ID                        |
| `node_id`         | PageIndex 节点 ID，页级降级检索时可为空   |
| `pages`           | PDF 页码范围                     |
| `option`          | A/B/C/D                      |
| `evidence_type`   | `support`、`refute`、`unclear` |
| `quote`           | 来自原文的短引文                     |
| `normalized_fact` | 面向题目的归一化事实                   |
| `numbers`         | 抽取出的金额、比例、期限、免赔额等            |
| `confidence`      | `high`、`medium`、`low`        |

每个选项都必须形成 verdict 记录，取值为 `support`、`refute` 或 `unclear`。被最终选中的选项必须有 `support` 证据；未选中的高风险选项应有 `refute` 证据，或记录补检索后仍为 `unclear` 的原因。

### 6.6 `CalculationEngine`

职责：

- 统一处理保险金额、比例、排序和扣减。
- 对题目中的万元、元、百分比、保单年度等单位做归一化。
- 将计算过程和结果写回证据记忆或答案判断上下文。

优先覆盖的计算类型：

- 身故保险金比较。
- 退保所得。
- 医疗费用扣除医保、免赔额后的赔付额。
- 多产品赔付排序。
- 给付比例和最高限额判断。

### 6.7 `AnswerJudge`

职责：

- 只基于证据记忆和程序计算结果逐项判断。
- 单选题输出唯一字母。
- 多选题输出去重、排序后的字母串。
- 判断题按题目选项含义输出 A 或 B。
- 对非法答案做程序化修正，不能把模型原始输出直接写入 `answer.csv`。

## 7. A榜 Insurance 端到端流程

### 7.1 离线预处理

1. 扫描 16 个保险 PDF。
2. 抽取每页文本，保存页级缓存。
3. 生成保险条款 Markdown 和页码映射。
4. 构建 Markdown PageIndex 主索引。
5. 生成 `node_spans`，把 Markdown 节点映射到 PDF 页码范围。
6. 运行 Markdown 主索引质量检查。
7. 仅对主索引不合格的文档构建 PDF PageIndex 兜底索引；实验性全量兜底构建必须单独标记。
8. 生成 A 榜必需的 `doc_catalog.jsonl`，同时为后续 B 榜预留。

### 7.2 在线答题

1. 读取 `insurance_questions.json`。
2. `QuestionParser` 结合 `doc_catalog.jsonl` 解析题目、选项、类型、产品别名和 `doc_ids`。
3. `DocRetriever` 直接返回 A 榜 `doc_ids`。
4. 对每个 `doc_id`，`InsuranceDocStore` 读取文档元数据和去掉 `text` 字段的紧凑树结构。
5. `TreeRetriever` 基于紧凑树结构选择相关节点，只产出节点、页码范围和取页理由。
6. `EvidenceExtractor` 按候选节点页码窗口调用 `InsuranceDocStore.get_page_content` 读取页级原文。
7. `EvidenceExtractor` 基于页级原文按选项生成 `support/refute/unclear` 证据记忆。
8. `CalculationEngine` 处理金额、比例和排序。
9. `AnswerJudge` 输出合法答案。
10. 写入 `answer.csv`、`evidence.jsonl` 和日志。

## 8. 保险领域检索策略

### 8.1 领域关键词

保险条款检索的高优先级关键词：

- 保险责任
- 责任免除
- 身故保险金
- 现金价值
- 账户价值
- 退保
- 犹豫期
- 等待期
- 宽限期
- 保单贷款
- 免赔额
- 给付比例
- 保险金申请
- 受益人
- 保险期间
- 释义

### 8.2 选项级检索

保险题经常是多产品比较或多选判断。系统不应只根据题干检索一次，而应把每个选项作为独立检索目标：

- 单个选项涉及一个产品时，优先检索该产品文档内的对应条款。
- 单个选项涉及多个产品时，拆成多个产品事实，再合并判断。
- 选项包含括号解释时，括号内容也参与关键词提取。
- 选项包含数值时，保留数值作为页内匹配和计算输入。

### 8.3 多产品比较

多产品题按“产品-字段”矩阵组织证据：

| 产品   | 字段    | 证据   | 计算值 | 判断    |
| ---- | ----- | ---- | --- | ----- |
| 产品 A | 身故保险金 | 原文页段 | 金额  | 支持/反驳 |
| 产品 B | 退保金额  | 原文页段 | 金额  | 支持/反驳 |

最终判断时先完成每个产品的事实抽取，再比较选项中的排序或条件，不让模型在缺少中间事实时直接猜答案。

## 9. 页段窗口与回退策略

### 9.1 硬性预算

默认预算参数：

| 参数                        | 默认值 | 说明                                        |
| ------------------------- | --- | ----------------------------------------- |
| `MAX_DOCS_PER_QUESTION`   | 4   | A 榜以题目 `doc_ids` 为准，超过时按题目顺序截断并记录 warning |
| `MAX_NODES_PER_DOC`       | 5   | 每个文档进入页读取的候选节点上限                          |
| `MAX_PAGES_PER_DOC`       | 8   | 每个文档读取的页数上限                               |
| `MAX_EVIDENCE_PER_OPTION` | 3   | 每个选项保留的证据上限                               |
| `MAX_RETRY_PER_QUESTION`  | 1   | 每题补检索或自检次数上限                              |

裁剪顺序：

1. 优先保留标题命中产品名、责任类别、金额/比例关键词的节点。
2. 优先保留页码范围更短的节点。
3. 多产品题按产品均衡保留节点，避免一个产品占满预算。
4. 已有 `support/refute` 高置信证据的选项不再继续扩展页段。

## 10. 模型与 Token 策略

正式推理问答阶段包括：

- 树节点选择。
- 页段证据抽取。
- 证据压缩。
- 选项判断。
- 答案自检。

这些阶段应使用 Qwen 系列模型，正式评测默认参考赛题基准模型 `Qwen3.6-plus`。所有调用通过统一模型客户端记录：

- `qid`
- `stage`
- `model`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `latency_ms`
- `success`
- `error`

`docs/模型配置.md` 中的 `ark-code-latest` 和 `ARK_API_KEY` 可用于开发调试、接口验证或本地实验，但正式赛题推理模型应以赛题要求的 Qwen 系列模型为准。架构中不得把 ARK 调试模型写成正式评测默认模型。

PageIndex 离线索引构建如调用 LLM，应单独记录构建日志。若索引构建在题目无关的离线预处理阶段完成，则不混入每题答题 Token；若构建或修复索引发生在正式答题阶段，则必须使用 Qwen 并纳入该题 Token 统计。

PageIndex 调用参数必须在项目侧配置中显式落盘，禁止隐式读取 PageIndex 默认 `config.yaml`。正式索引构建推荐参数：

| 参数                       | Markdown 主路 | PDF 兜底路        |
| ------------------------ | ----------- | -------------- |
| `model`                  | 正式 Qwen 模型  | 正式 Qwen 模型     |
| `if_add_node_id`         | `yes`       | `yes`          |
| `if_add_node_summary`    | `no`        | `no`           |
| `if_add_doc_description` | `no`        | `no`           |
| `if_add_node_text`       | `no`        | `no`           |
| `toc_check_page_num`     | 不适用         | 显式配置，默认 20     |
| `max_page_num_each_node` | 不适用         | 显式配置，建议 6 到 10 |

开发阶段可以用非正式模型做本地实验，但其生成的 PageIndex 结构、摘要、选择结果或纠错结果不得进入正式提交链路；正式提交所用索引若包含 LLM 生成的结构信息，应使用赛题允许的 Qwen 模型构建并保留日志。

## 11. 输出格式

### 11.1 `answer.csv`

字段：

| 字段                  | 说明                    |
| ------------------- | --------------------- |
| `qid`               | 题目 ID，汇总行使用 `summary` |
| `answer`            | 合法答案字母                |
| `prompt_tokens`     | 该题输入 Token            |
| `completion_tokens` | 该题输出 Token            |
| `total_tokens`      | 两者之和                  |

多选题答案必须去重、排序，不使用空格、逗号或其他分隔符。

### 11.2 `evidence.jsonl`

每题一行，建议包含：

- `qid`
- `answer`
- `candidate_docs`
- `selected_nodes`
- `evidence`
- `calculations`
- `usage`
- `fallbacks`
- `warnings`

`evidence` 内每条证据必须能追溯到 `doc_id + node_id/pages + quote`。

`evidence.jsonl` 还应包含 `option_judgements`，记录每个选项的 verdict、证据 ID、计算 ID 和补检索状态。计算/排序题的 `calculations` 必须保留每个产品字段的中间事实、归一化数值、单位和最终比较结果。

### 11.3 日志

日志保存到 `outputs/insurance_a/logs/`。

日志不得保存 API Key，不得引入标准答案或外部答案信息。

## 12. 扩展设计

### 12.1 扩展到 B 榜

B 榜缺少 `doc_ids`，扩展时只替换 `DocRetriever`：

- 当前 A 榜：`DocRetriever(question) -> question.doc_ids`
- B 榜：`DocRetriever(question, doc_catalog) -> candidate_doc_ids`

其余模块继续复用：

- `InsuranceDocStore`
- `TreeRetriever`
- `EvidenceExtractor`
- `CalculationEngine`
- `AnswerJudge`

B 榜候选文档召回应基于 `domain`、元数据、标题、章节标题、关键词和 Qwen 候选选择，不使用非 Qwen 模型生成的向量、重排或语义摘要参与正式答题。

### 12.2 扩展到其他领域

可复用模块：

- 文档解析层。
- PageIndex 结构索引层。
- 通用 `DocStore`。
- 通用树检索。
- 证据记忆结构。
- 答案合法化。
- Token 统计和输出审计。

领域专属模块：

| 领域                    | 需要替换或增强的部分                  |
| --------------------- | --------------------------- |
| `regulatory`          | 法条编号、义务主体、期限、处罚、适用范围识别。     |
| `financial_contracts` | 债券发行条款、评级、担保、募集资金用途识别。      |
| `financial_reports`   | 财务指标、年份、公司主体、表格和同口径计算。      |
| `research`            | 研报标题、图表标题、预测指标、观点核验。        |
| `insurance`           | 产品条款、责任触发、免责、赔付公式、免赔额和现金价值。 |

领域扩展应新增 `DomainAdapter`，而不是改动 PageIndex 封装和输出审计层。

## 13. 验收标准

### 13.1 数据完整性

- 16 个 insurance PDF 均生成页级文本缓存。
- 每个文档均尝试生成 Markdown 主索引；无法生成或质量不合格时必须记录原因，并生成 PDF 兜底索引或页级降级标记。
- 每个 Markdown 主索引都有对应 `node_spans`，且所有可用节点能映射到 PDF 页码范围。
- `doc_catalog.jsonl` 覆盖 16 个 `doc_id`，并能映射 A 榜题目中出现的产品名和别名。
- 质量报告能标记 Markdown 主路、PDF 兜底或页级降级状态。
- 质量报告至少包含节点数、空标题数、坏页码范围数、关键词标题命中数、页码映射覆盖率。

### 13.2 A 榜链路

- 20 道 `ins_a_*` 题能完整走通。
- 每题直接使用题目 `doc_ids`。
- 不启用 B 榜候选召回。
- 题目中全部 `doc_ids` 都能在 `doc_catalog.jsonl` 中找到产品元数据。

### 13.3 证据可追溯

- 每个选项都有 `support/refute/unclear` verdict 记录。
- 每个被最终选中的选项都必须有支持证据，且证据可追溯到 `doc_id + node_id/pages + quote`。
- 未选中的高风险选项应保留反驳证据；若补检索后仍无法判断，必须记录 `unclear` 原因。
- 计算题和排序题必须保留每个产品字段的中间事实、归一化数值、单位和计算结果。

### 13.4 答案格式

- `mcq` 输出一个大写字母。
- `multi` 输出排序后的大写字母串。
- `tf` 输出 A 或 B。
- 无非法字符、空格、逗号或重复字母。

### 13.5 回退策略

- Markdown 失败时能切换 PDF 兜底。
- 节点定位不足时能扩大页窗口。
- 证据不足时能触发一次补检索。
- 回退动作写入 `evidence.jsonl` 或日志。
- PDF 兜底索引按质量触发；若全量构建，索引日志必须标记为实验性全量兜底。

### 13.6 扩展性

- 文档中明确 A 榜 `DocRetriever` 是 passthrough。
- B 榜候选召回作为扩展接口存在，但不是当前主流程。
- insurance 专属逻辑集中在 `DomainAdapter`、关键词策略和计算规则中。
- PageIndex 封装、证据记忆、输出审计可以被其他领域复用。

## 14. 关键结论

当前 A 榜 insurance 系统应采用确定性流水线，而不是让 Agent 自由浏览全文。PageIndex 的职责是提供结构化章节树和可追溯节点定位；项目侧围绕题目和选项完成保险领域检索、证据记忆、程序计算和答案约束。

双路线索引是本设计的核心：Markdown 优先保证保险条款结构和页码映射质量，PDF 直连 PageIndex 作为快速验证和兜底。这样既能服务当前 20 道 A 榜保险题，也能在后续扩展到 B 榜和其他金融文档领域时保持模块边界稳定。
