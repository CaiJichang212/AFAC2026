# 金融Agent问答系统技术调研与模型调用报告

版本：v1.0  
日期：2026-06-13  
调研范围：Qwen API、文档解析、RAG（检索增强生成）、Agent（智能体）、动态记忆压缩、混合检索、开源项目与学术论文。  
信息源原则：优先采用官方文档、官方 GitHub、论文原文或权威会议论文；博客仅用于补充，不作为核心依据。

## 1. 赛题技术约束摘要

正式推理问答阶段必须使用 Qwen 系列模型 API。文档解析阶段可以使用非 Qwen 工具进行 OCR、版面分析、表格恢复、阅读顺序还原和 PDF 转结构化文本，但这些工具的输出只能作为可读文本、表格或版面结构，不得形成非 Qwen 语义向量、非 Qwen rerank（重排序）、非 Qwen 语义摘要、FAQ 或结论参与正式答题。

这意味着系统架构要把“预处理”和“正式推理”严格隔离。预处理产物可以包括页面文本、表格 HTML、章节层级、坐标、图片路径、页码和文件哈希；正式推理产物必须来自规则检索、符号索引和 Qwen 系列模型调用。

## 2. Qwen API 与调用能力调研

### 2.1 官方调用入口

Alibaba Cloud Model Studio 官方文档说明 Qwen 模型提供多种接口，包括 OpenAI Chat Completion、OpenAI Responses 和 DashScope 原生接口。OpenAI-compatible API 只需调整 API Key、BASE_URL 与模型名即可迁移现有 OpenAI SDK 代码。[^qwen-api][^qwen-openai]

推荐调用形态如下。

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[
        {"role": "system", "content": "你是金融长文档问答证据核验器。"},
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    # 部分 SDK 需要把 Qwen 扩展参数放入 extra_body，具体以百炼/Model Studio SDK 版本为准。
    extra_body={"enable_thinking": False},
)
usage = resp.usage
```

### 2.2 qwen3.6-plus 适配性

Model Studio 文档建议在聊天机器人、内容生成、摘要和文档处理场景使用 `qwen3.6-plus`，并说明其提供 1M 上下文窗口、思考模式、函数调用、内置工具、结构化输出和批量调用等能力。[^qwen36] 赛题基准模型指定为 Qwen3.6-plus，因此技术方案以该模型为主。

对本赛题的含义：

- 1M 上下文可作为兜底，但不应默认全文输入，否则 TokenScore（Token 效率分）会下降。
- 思考模式（Thinking Mode）适合复杂法规推理、数值计算公式选择和多文档冲突分析；普通证据判断可关闭以降低输出 Token。
- 结构化输出适合逐选项判断 JSON；但官方文档提示部分 thinking 模式不支持结构化输出，因此建议“证据判断 JSON 输出”默认使用非 thinking 模式，复杂题再单独启用深度思考并做 JSON 修复。[^qwen36][^qwen-openai]
- Batch Inference（批量推理）适合高吞吐、非低延迟任务，可用于正式评测前的全量离线作答。

### 2.3 Qwen Embedding 的合规边界

Model Studio Embedding 文档说明 `text-embedding-v4` 可通过 OpenAI-compatible API 调用，并且模型列表中标注其属于 Qwen3-Embedding 系列，支持多语言与最多 8,192 tokens per text 的输入。[^qwen-embedding] 但赛题规则的关键表述是“不得使用非 Qwen 模型生成的向量”。因此建议实现两个检索配置：

- `strict_competition`：默认关闭向量检索，只使用 BM25、短语、正则、字段、数值、条款编号索引，以及 Qwen LLM 证据判断。
- `qwen_embedding_allowed`：若赛题方确认 `text-embedding-v4` 属于允许范围，则启用 Qwen Embedding 与稀疏检索融合。

### 2.4 Context Cache（上下文缓存）

Model Studio Context Cache 文档说明，缓存可复用共同输入前缀以降低推理延迟和费用；显式缓存命中按 10% 标准输入价计费，隐式缓存命中按 20% 标准输入价计费，并且不影响输出质量。[^qwen36] 但提交要求的 Token 统计是模型调用过程中的 `prompt_tokens`、`completion_tokens` 和 `total_tokens`，不应将计费折扣等同于评测 Token 消耗降低。缓存可用于降低成本和加速，但不能作为减少 `summary.total_tokens` 的统计手段。

### 2.5 Qwen-Agent 与 Agent 开发

Qwen-Agent 官方仓库说明该框架用于基于 Qwen 的指令遵循、工具使用、规划和记忆能力构建 LLM 应用。[^qwen-agent] Qwen3.6 官方仓库也指向 Qwen-Agent 作为 Agent 开发框架。[^qwen36-github]

对本赛题的建议：可以借鉴 Qwen-Agent 的工具编排思想，但正式系统不必强依赖完整框架。赛题任务较短、评测严格，建议实现轻量 Agent Orchestrator（智能体编排器），只开放有限工具：检索、证据扩展、计算器、答案校验、Token 统计。

## 3. 文档解析技术调研

### 3.1 MinerU

MinerU 官方仓库说明其可将 PDF、图片、DOCX、PPTX、XLSX 转为 Markdown/JSON，支持阅读顺序还原、标题段落列表结构保留、图片/表格/表题/脚注提取、公式转 LaTeX、表格转 HTML、扫描 PDF 和乱码 PDF OCR、多语言 OCR、可视化结果和 CLI/FastAPI/Gradio 部署。[^mineru]

适用场景：财报、研报、合同等复杂 PDF 的结构化解析。  
风险：MinerU 若使用非 Qwen 模型，只能用于解析，不得让其生成语义摘要或向量。  
建议：正式流程中保留 MinerU 生成的 Markdown、表格 HTML、页码、bbox，但不要使用其语义摘要。

### 3.2 PyMuPDF

PyMuPDF 官方文档说明它是高性能 Python 库，可用于 PDF 等文档的数据抽取、分析、转换和操作。[^pymupdf] 对文本型 PDF，PyMuPDF 的速度优势明显，适合做基础抽取、页码保留、文本块和图片导出。

适用场景：年报、募集说明书、保险条款、研报的全文抽取与页级切分。  
风险：复杂表格结构恢复能力有限。  
建议：作为第一解析器，生成页级纯文本与块级坐标。

### 3.3 pdfplumber

pdfplumber 官方仓库说明其可获取 PDF 中每个字符、矩形、线条等详细信息，并支持表格抽取和可视化调试，适合机器生成 PDF。[^pdfplumber]

适用场景：财报表格、债券条款表、发行摘要表。  
风险：扫描 PDF 和复杂视觉表格需结合 OCR 或其他工具。  
建议：用于表格抽取和解析质量诊断，与 PyMuPDF 结果交叉校验。

### 3.4 PaddleOCR

PaddleOCR 官方仓库说明其可将 PDF 和图片转换为结构化、LLM-ready 的 JSON/Markdown，支持 100+ 语言，并包含文档解析能力。[^paddleocr]

适用场景：扫描件、图片型 PDF、印章/手写/低质量页面。  
风险：属于非 Qwen 预处理工具，不能参与正式语义推理。  
建议：只在 PyMuPDF 抽取文本过少或检测为扫描页时启用。

### 3.5 Camelot

Camelot 文档说明其用于从文本型 PDF 提取表格，可导出 CSV、JSON、Excel、HTML、Markdown 等格式，并提供准确率和空白率等解析指标；同时说明它只适用于文本型 PDF，不适用于扫描文档。[^camelot]

适用场景：规则表格、财务报表表格、募集说明书发行要素表。  
风险：扫描页不可用，跨页复杂表格需要补充逻辑。  
建议：表格解析质量高时优先采用；低质量表格回退 pdfplumber/MinerU。

## 4. RAG、检索与排序技术调研

### 4.1 RAG 基础

RAG（Retrieval-Augmented Generation，检索增强生成）论文提出将参数化模型记忆与非参数化外部记忆结合，使生成模型能访问检索到的证据并改善知识密集型任务的事实性。[^rag] 本赛题的核心正是基于给定金融长文档的知识密集型问答，因此 RAG 是基础框架。

### 4.2 DPR 与稠密检索

DPR（Dense Passage Retrieval，稠密段落检索）使用双编码器学习问题和段落向量，在开放域问答中优于强 BM25 基线。[^dpr] 但本赛题存在合规限制：非 Qwen 稠密向量不得参与正式答题。因此 DPR 思路可借鉴，但正式实现只能使用 Qwen Embedding 或规则/稀疏索引。

### 4.3 FiD 与多证据融合

FiD（Fusion-in-Decoder，解码器融合）研究如何让生成模型利用多个检索段落进行开放域问答。[^fid] 对本赛题启示是：最终判断不应只看 Top-1 证据，而应保留多个文档、多个段落、多个表格块，并让模型在证据集合上逐选项判断。

### 4.4 BM25、RRF 与混合检索

BM25（Best Matching 25，最佳匹配25）适合条款编号、机构名、指标名、年份、金额等精确匹配。RRF（Reciprocal Rank Fusion，倒数排序融合）官方文档说明可将多个不同相关性指标的结果集融合为一个结果集，且不要求不同指标相关。[^elastic-rrf]

建议正式系统使用：

- BM25 正文召回。
- 标题/章节字段加权召回。
- 短语匹配：公司名、产品名、法规名、指标名。
- 正则召回：金额、比例、日期、条款编号。
- RRF 融合：题干查询、选项查询、实体查询、数值查询。
- Qwen LLM rerank：只对 Top-N 片段做合规重排或证据判定。

### 4.5 Faiss 与 Milvus

Faiss 官方文档说明其用于高效相似度搜索和稠密向量聚类，支持大规模向量集合。[^faiss] Milvus 文档说明混合检索可组合 Dense Vector（稠密向量）与 Sparse Vector（稀疏向量），兼顾语义关系和精确关键词匹配。[^milvus-hybrid]

合规建议：这两类工具可作为向量索引基础设施，但正式模式下只有当向量由 Qwen 系列 Embedding 生成且赛题方认可时才启用。否则，Faiss/Milvus 应禁用或仅用于调试。

## 5. 开源 RAG/Agent 项目调研

| 项目 | 主要能力 | 对本赛题可借鉴点 | 正式使用建议 |
|---|---|---|---|
| Qwen-Agent | Qwen 原生 Agent 框架，支持工具、规划、记忆 | 工具编排、Qwen 调用封装 | 可借鉴或轻量集成 |
| RAGFlow | 基于深度文档理解的开源 RAG 引擎，强调复杂格式数据和引用 | 文档解析、知识库与引用设计 | 可借鉴流程；正式推理需合规改造 |
| LlamaIndex | 数据连接器、索引、检索、Query Engine 等 | 数据结构设计、索引抽象 | 不直接用非 Qwen 模型组件 |
| Haystack | 生产级 RAG、Agent、多模态搜索管线 | Pipeline 组合、组件化工程 | 可借鉴架构思想 |
| MinerU | PDF/Office 到 Markdown/JSON 解析 | 预处理解析主力候选 | 可用于预处理，不用于推理 |

RAGFlow 文档强调基于深度文档理解的 RAG，可从复杂格式数据中提供带引用的问答能力。[^ragflow] LlamaIndex 官方仓库强调数据接入、结构化索引、检索/查询接口和可定制组件。[^llamaindex] Haystack 文档强调生产级 AI Agent、RAG 和多模态搜索管线。[^haystack]

## 6. 动态记忆压缩相关论文

### 6.1 LLMLingua / LongLLMLingua

LLMLingua 提出粗到细 Prompt 压缩，以较高压缩率减少推理成本并保持语义完整。[^llmlingua] LongLLMLingua 面向长上下文场景，指出长上下文 LLM 面临成本高、性能下降和位置偏置问题，并通过问题感知压缩提升关键信息密度。[^longllmlingua]

对本赛题启示：不要把检索片段机械拼接给模型，而应做问题感知压缩。但由于正式问答只能使用 Qwen，压缩器也必须是 Qwen 或规则抽取。最佳实践是：保留原文证据 + 提取结构化事实，最终 Prompt 同时包含短原文与结构化记忆。

### 6.2 RAPTOR

RAPTOR 提出递归地对文本块进行嵌入、聚类和摘要，构建树状检索结构，在推理时从不同抽象层次检索信息。[^raptor]

对本赛题启示：合同和年报可构建章节树、页树和表格树。但正式合规风险在于：如果摘要由非 Qwen 生成则不能用于正式答题。可采用规则章节树，或使用 Qwen 生成章节级“索引摘要”并记录 Token；在 strict 模式中优先使用标题与目录而非语义摘要。

### 6.3 ReAct 与 Self-RAG

ReAct 提出让模型交替产生推理轨迹和动作，以便通过外部工具获取信息并提升可解释性。[^react] Self-RAG 研究让模型按需检索、生成并自我反思，从而改善事实性和引用质量。[^selfrag]

对本赛题启示：Agent 不应一次性完成所有工作，而应按“检索不足 -> 扩展证据 -> 再判断”的循环运行。但由于 Token 预算有限，不建议对所有题做多轮反思；只对低置信、证据冲突、答案为空或多选全选/全不选的题触发二次检索与自检。

## 7. 推荐调用链路

### 7.1 模型调用类型

| 调用类型 | 是否必需 | 模型 | 输入 | 输出 | Token 控制 |
|---|---|---|---|---|---|
| Question Parser | 可选 | Qwen3.6-plus / qwen-flash | 题干+选项 | 结构化任务 JSON | 规则能解析时不调用 |
| Evidence Compressor | 必需 | Qwen3.6-plus | Top-K 原文证据 | 结构化证据记忆 | 每题每轮不超过 3,000-5,000 输入 Token |
| Option Judge | 必需 | Qwen3.6-plus | 题目+选项+证据 | 逐选项 JSON | 只给相关证据，不给全文 |
| Self Check | 条件触发 | Qwen3.6-plus thinking | 冲突题/低置信题 | 修正建议 JSON | 只对 20%-40% 题触发 |
| Answer Formatter | 不调用 | 程序 | 判断结果 | 合法字母 | 零 Token |

### 7.2 结构化输出 Schema

```json
{
  "qid": "fin_a_001",
  "option_judgements": [
    {
      "option": "A",
      "verdict": "true|false|unknown",
      "confidence": 0.0,
      "evidence_ids": ["e1", "e2"],
      "calculation": "可为空；如涉及数值，写公式和标准化数值",
      "short_reason": "一句话说明"
    }
  ],
  "final_answer": "AC",
  "need_more_evidence": false
}
```

### 7.3 Prompt 模板建议

系统提示词应强调：只基于证据回答；不要使用常识；无法由证据支持则判 false 或 unknown；输出严格 JSON；不要输出与 JSON 无关的文字。

用户提示词建议结构：

```text
# 任务
你需要判断金融问答题中每个选项是否被证据支持。

# 题目信息
qid: ...
domain: ...
answer_format: multi
question: ...
options: ...

# 证据
[e1] doc_id=..., page=..., section=...
原文：...
结构化事实：...

# 输出要求
返回 JSON。multi 题 final_answer 为所有 true 选项按字母升序拼接；mcq/tf 只能一个字母。
```

### 7.4 Token 统计

每次 API 响应都应读取 `usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens`。如果响应包含缓存命中细节，也应单独记录，但提交 `answer.csv` 的 token 字段应按 `usage` 汇总。

```python
token_meter.add(
    qid=qid,
    call_type="option_judge",
    prompt_tokens=resp.usage.prompt_tokens,
    completion_tokens=resp.usage.completion_tokens,
    total_tokens=resp.usage.total_tokens,
    model="qwen3.6-plus",
)
```

## 8. 合规实现建议

正式 `strict_competition` 配置：

```yaml
models:
  judge: qwen3.6-plus
  compressor: qwen3.6-plus
  self_check: qwen3.6-plus
retrieval:
  lexical_bm25: true
  regex_index: true
  table_index: true
  qwen_embedding: false
  non_qwen_embedding: false
  non_qwen_rerank: false
preprocess:
  mineru: true
  pymupdf: true
  pdfplumber: true
  paddleocr_on_scanned_pages: true
  save_semantic_summary: false
logging:
  save_evidence: true
  save_model_usage: true
```

可选 `qwen_embedding_allowed` 配置只有在赛题方确认 Qwen Embedding 合规时启用：

```yaml
retrieval:
  qwen_embedding: true
  embedding_model: text-embedding-v4
  vector_store: faiss
  fusion: rrf
```

## 9. 技术选型结论

推荐主线方案：

1. 文档预处理：PyMuPDF 页级文本 + MinerU 结构化 Markdown/JSON + pdfplumber/Camelot 表格校验 + PaddleOCR 扫描页兜底。
2. 索引：BM25、标题/章节字段索引、正则数值索引、条款编号索引、表格单元格索引；可选 Qwen Embedding。
3. Agent：轻量 orchestrator，控制检索、扩展、压缩、计算、判断、自检和输出。
4. 记忆压缩：原文证据不丢，Qwen 生成结构化 `EvidenceMemory`，最终 Prompt 控制在题目难度对应预算。
5. 推理：Qwen3.6-plus 逐选项判断，复杂题开启 thinking 或二次自检。
6. 后处理：程序确定性生成答案字母、Token 汇总和证据文件。

该方案兼顾准确率、合规性、可复现性和 Token 成本，适合在 A 榜快速闭环，并能扩展到 B 榜无 `doc_ids` 的候选文档定位场景。

## 10. 参考资料

[^qwen-api]: Alibaba Cloud Model Studio, “Qwen API reference”, https://www.alibabacloud.com/help/en/model-studio/qwen-api-reference/ ，访问日期：2026-06-13。
[^qwen-openai]: Alibaba Cloud Model Studio, “OpenAI compatible - Chat”, https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope ，访问日期：2026-06-13。
[^qwen36]: Alibaba Cloud Model Studio, “Text generation models”, https://www.alibabacloud.com/help/en/model-studio/text-generation-model ，访问日期：2026-06-13。
[^qwen-embedding]: Alibaba Cloud Model Studio, “Embedding”, https://www.alibabacloud.com/help/en/model-studio/embedding ，访问日期：2026-06-13。
[^qwen-agent]: QwenLM/Qwen-Agent, GitHub, https://github.com/QwenLM/Qwen-Agent ，访问日期：2026-06-13。
[^qwen36-github]: QwenLM/Qwen3.6, GitHub, https://github.com/QwenLM/Qwen3.6 ，访问日期：2026-06-13。
[^mineru]: OpenDataLab/MinerU, GitHub, https://github.com/opendatalab/mineru ，访问日期：2026-06-13。
[^ragflow]: RAGFlow Docs, https://ragflow.io/docs/ ，访问日期：2026-06-13。
[^pymupdf]: PyMuPDF Documentation, https://pymupdf.readthedocs.io/ ，访问日期：2026-06-13。
[^pdfplumber]: jsvine/pdfplumber, GitHub, https://github.com/jsvine/pdfplumber ，访问日期：2026-06-13。
[^paddleocr]: PaddlePaddle/PaddleOCR, GitHub, https://github.com/PaddlePaddle/PaddleOCR ，访问日期：2026-06-13。
[^camelot]: Camelot Documentation, https://camelot-py.readthedocs.io/ ，访问日期：2026-06-13。
[^llamaindex]: run-llama/llama_index, GitHub, https://github.com/run-llama/llama_index ，访问日期：2026-06-13。
[^haystack]: Haystack Documentation, https://docs.haystack.deepset.ai/docs/intro ，访问日期：2026-06-13。
[^elastic-rrf]: Elastic, “Reciprocal rank fusion”, https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion ，访问日期：2026-06-13。
[^faiss]: Faiss Documentation, https://faiss.ai/index.html ，访问日期：2026-06-13。
[^milvus-hybrid]: Milvus Documentation, “Multi-Vector Hybrid Search”, https://milvus.io/docs/multi-vector-search.md ，访问日期：2026-06-13。
[^rag]: Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”, NeurIPS 2020, https://arxiv.org/abs/2005.11401 。
[^dpr]: Karpukhin et al., “Dense Passage Retrieval for Open-Domain Question Answering”, 2020, https://arxiv.org/abs/2004.04906 。
[^fid]: Izacard and Grave, “Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering”, 2020, https://arxiv.org/abs/2007.01282 。
[^react]: Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models”, 2022, https://arxiv.org/abs/2210.03629 。
[^selfrag]: Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection”, 2023, https://arxiv.org/abs/2310.11511 。
[^llmlingua]: Jiang et al., “LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models”, 2023, https://arxiv.org/abs/2310.05736 。
[^longllmlingua]: Jiang et al., “LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression”, 2023, https://arxiv.org/abs/2310.06839 。
[^raptor]: Sarthi et al., “RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval”, 2024, https://arxiv.org/abs/2401.18059 。
