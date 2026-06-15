# PageIndex 源码解读：面向“金融长文本 Agent 问答”赛题

本文档面向已经将 PageIndex 作为 `git submodule` 管理的赛题项目。目标不是改 PageIndex 源码，而是把它作为第三方库复用：生成文档树索引、读取树结构、按页读取证据文本，并在项目侧实现金融问答 Agent、候选文档检索、证据聚合、答案约束和 Token 统计。

## 1. 结论先行

PageIndex 与本赛题最相关的能力是“结构化长文档索引 + 基于树结构的可追溯检索”。它适合解决金融长文本中“章节层级清晰、证据分散、题目需要定位具体条款/指标/表格”的问题。对赛题而言，PageIndex 不应被当作完整问答系统直接使用，而应被放在检索与上下文管理层：先把 PDF 或清洗后的 Markdown 转成树，再由赛题 Agent 基于题目和选项在树中选择节点、读取紧凑页段、抽取证据、逐项判断。

推荐复用的源码模块是：

- `pageindex.page_index.page_index()`：PDF 树索引生成入口，适合直接调用并传入赛题自定义参数。
- `pageindex.page_index_md.md_to_tree()`：Markdown 树索引生成入口，适合接入 MinerU、版面分析、OCR、PDF 转 Markdown 等预处理结果。
- `pageindex.retrieve.get_document_structure()` 与 `get_page_content()`：作为 Agent 的工具函数，分别返回去掉正文的树结构和指定页/行内容。
- `pageindex.utils.structure_to_list()`、`create_node_mapping()`、`remove_fields()`、`count_tokens()` 等工具函数：适合构建项目侧的节点检索、节点映射、Token 估算和上下文裁剪。

不建议直接复用 `PageIndexClient.index()` 作为赛题索引构建主入口。该方法在当前源码中对 PDF 与 Markdown 都硬编码启用了 `if_add_node_summary='yes'`、`if_add_doc_description='yes'`、`if_add_node_text='yes'`。这会增加索引成本和存储体积，也可能触碰赛题对非 Qwen 模型语义摘要参与正式答题的限制。更稳妥的方式是在项目侧写一个薄封装，直接调用 `page_index()` 或 `md_to_tree()` 并显式关闭或控制摘要、正文字段。

## 2. 赛题相关背景抽象

本赛题的核心不是单段阅读，而是多文档、长上下文、证据定位、数值/条件推理和答案规范化。A 榜通常给出 `doc_ids`，B 榜不直接给 `doc_ids`，需要先做候选文档检索。题型包括单选、多选和判断，最终提交必须是严格字母答案，并带 Token 统计。

PageIndex 解决的是“给定候选文档后，如何更低成本、更可解释地找到证据区域”。它不直接解决以下问题：

- B 榜跨文档候选召回。
- 金融题型解析、选项拆解、数值计算。
- 多文档证据冲突消解。
- 最终答案字母规范化。
- 评测所需的 Token 用量统计。

因此，项目侧需要在 PageIndex 之上补齐金融 Agent 层。

## 3. 源码目录与职责

当前 PageIndex 源码的核心文件如下：

```text
PageIndex/
├── pageindex/
│   ├── __init__.py
│   ├── client.py
│   ├── config.yaml
│   ├── page_index.py
│   ├── page_index_md.py
│   ├── retrieve.py
│   └── utils.py
├── examples/
│   └── agentic_vectorless_rag_demo.py
├── examples/tutorials/
│   ├── doc-search/
│   └── tree-search/
├── run_pageindex.py
└── requirements.txt
```

各文件与赛题相关性如下：

| 文件 | 作用 | 赛题相关性 | 建议 |
| --- | --- | --- | --- |
| `pageindex/page_index.py` | PDF 树索引构建。包含目录检测、目录抽取、目录页码映射、无目录文档树生成、索引校验、节点摘要生成等逻辑。 | 高 | 直接调用底层 `page_index()`，不要依赖默认配置。 |
| `pageindex/page_index_md.py` | Markdown 树索引构建。根据 `#` 层级生成树，可选生成节点摘要、文档描述、节点正文。 | 高 | 对金融 PDF 先做高质量解析为 Markdown 时优先使用。 |
| `pageindex/retrieve.py` | 工具函数：返回文档元信息、无正文树结构、指定页/行内容。 | 高 | 适合作为 Agent tools 或服务接口。 |
| `pageindex/utils.py` | LLM 调用、Token 计数、JSON 抽取、树遍历、PDF 文本抽取、节点正文/摘要生成、配置加载等工具。 | 高 | 复用树处理函数；LLM 调用建议由项目侧统一封装或 monkeypatch 统计。 |
| `pageindex/client.py` | 高层客户端：索引 PDF/MD、维护 workspace、提供读取工具。 | 中 | 可借鉴 workspace 设计；不建议直接用 `index()` 构建正式赛题索引。 |
| `run_pageindex.py` | 命令行生成 PDF/MD 树索引。 | 中 | 可作为离线索引脚本参考。 |
| `examples/agentic_vectorless_rag_demo.py` | OpenAI Agents SDK 示例，演示三类工具和自动工具调用。 | 中 | 借鉴工具设计，不直接用于赛题。 |
| `examples/tutorials/doc-search` | 多文档检索建议：元数据、描述、语义检索。 | 中 | 赛题 B 榜优先采用元数据/规则/关键词/Qwen 选择；避免非 Qwen 向量召回。 |
| `examples/tutorials/tree-search` | 给出基于树结构让 LLM 选择相关 node_id 的提示词模式。 | 高 | 可改写为 Qwen 树检索 prompt。 |

## 4. PageIndex 的核心数据结构

### 4.1 PDF 树结构

`page_index()` 对 PDF 的典型输出是：

```json
{
  "doc_name": "xxx.pdf",
  "doc_description": "可选的一句话文档描述",
  "structure": [
    {
      "title": "章节标题",
      "node_id": "0001",
      "start_index": 12,
      "end_index": 18,
      "summary": "可选的节点摘要",
      "text": "可选的节点正文",
      "nodes": [
        {
          "title": "子章节标题",
          "node_id": "0002",
          "start_index": 13,
          "end_index": 15
        }
      ]
    }
  ]
}
```

对赛题最重要的是：

- `title`：章节标题，是树检索的主要结构信号。
- `node_id`：节点稳定引用 ID，可用于证据追踪。
- `start_index` / `end_index`：PDF 物理页码范围，适合调用 `get_page_content()` 读取原文。
- `summary`：节点摘要。是否保留取决于是否使用 Qwen 生成，以及是否愿意为摘要成本买单。
- `text`：节点正文。通常不应放入树结构返回给 LLM，否则会抵消 PageIndex 的 Token 优势。正文应在确定页段后再读取。

### 4.2 Markdown 树结构

`md_to_tree()` 对 Markdown 的典型输出是：

```json
{
  "doc_name": "xxx",
  "line_count": 1234,
  "doc_description": "可选",
  "structure": [
    {
      "title": "一级标题",
      "node_id": "0001",
      "line_num": 10,
      "summary": "可选",
      "prefix_summary": "可选",
      "text": "可选",
      "nodes": []
    }
  ]
}
```

Markdown 模式使用标题行的 `#` 数量确定层级，用 `line_num` 定位内容。若金融 PDF 已通过 OCR/版面分析恢复为结构化 Markdown，Markdown 模式通常比 PyPDF2 直接抽 PDF 更稳定，尤其适用于法规条款、年报标题、研报章节等结构较强的文本。

### 4.3 PageIndexClient workspace 存储

`PageIndexClient` 在指定 `workspace` 时，会将每个文档保存成 `{uuid}.json`，并维护 `_meta.json`。其设计对赛题可借鉴，但有两个注意点：

1. `PageIndexClient.index()` 生成的是随机 UUID，不是赛题原始 `doc_id`。正式系统需要维护 `competition_doc_id -> pageindex_doc_id` 映射，或者绕开 `PageIndexClient.index()`，用赛题 `doc_id` 作为索引文件名。
2. PDF 保存时 `_save_doc()` 会从 `structure` 中移除 `text` 字段，以避免正文重复存储；正文存在 `pages` 缓存中。这一点符合赛题按需取页的设计。

## 5. PDF 索引构建源码流程

`pageindex/page_index.py` 是 PageIndex PDF 处理的主文件。其主流程可以概括为：

```text
page_index()
  -> ConfigLoader().load(user_opt)
  -> page_index_main(doc, opt)
       -> get_page_tokens()                    # PyPDF2/PyMuPDF 抽页文本并计 token
       -> tree_parser()
            -> check_toc()                     # 检测目录页
                 -> find_toc_pages()
                 -> toc_extractor()
                 -> detect_page_index()
            -> meta_processor()                # 根据目录情况选择处理模式
                 -> process_toc_with_page_numbers()
                 -> process_toc_no_page_numbers()
                 -> process_no_toc()
            -> add_preface_if_needed()
            -> check_title_appearance_in_start_concurrent()
            -> post_processing()
            -> process_large_node_recursively()
       -> write_node_id()
       -> add_node_text()                      # 可选
       -> generate_summaries_for_structure()   # 可选
       -> generate_doc_description()           # 可选
       -> format_structure()
```

### 5.1 三种目录处理模式

PageIndex 会先尝试在前若干页检测目录。默认配置中 `toc_check_page_num=20`。根据文档是否有目录、目录是否包含页码，进入不同路径。

| 模式 | 触发条件 | 核心函数 | 作用 |
| --- | --- | --- | --- |
| `process_toc_with_page_numbers` | 检测到目录，且目录中包含页码 | `toc_transformer()`、`toc_index_extractor()`、`calculate_page_offset()` | 把目录转成 JSON，计算目录页码与 PDF 物理页之间的偏移，得到物理页范围。 |
| `process_toc_no_page_numbers` | 检测到目录，但目录无页码 | `toc_transformer()`、`add_page_number_to_toc()` | 先把目录转 JSON，再扫描正文寻找各标题的物理页。 |
| `process_no_toc` | 未检测到目录或前两种路径失败 | `generate_toc_init()`、`generate_toc_continue()` | 直接从正文中抽取层级结构和标题所在页。 |

金融文档中常见的目录页、条款目录、年报目录、研报目录都适合第一类或第二类路径。保险条款、法规文件有时没有标准目录，此时会走 `process_no_toc`。

### 5.2 页码偏移与物理页

金融 PDF 经常存在封面、目录、公告页、页脚页码与 PDF 物理页不一致的问题。PageIndex 的处理方式是：

1. 从目录中读取章节标题和目录页码。
2. 在正文开始部分寻找标题真正出现的物理页。
3. 计算 `physical_index - page` 的众数作为页码偏移。
4. 把目录页码转成 PDF 物理页码。

这对年报、债券募集说明书、监管规则汇编等文档很有价值，因为赛题证据最终需要定位到 PDF 页或清洗后文本位置。

### 5.3 索引校验与修复

PageIndex 在生成初步 TOC 后会校验标题是否出现在预期页：

- `verify_toc()`：并发检查每个标题是否出现在对应页。
- `fix_incorrect_toc()`：对错误标题，在相邻正确节点之间重新寻找物理页。
- `fix_incorrect_toc_with_retries()`：多轮修复。
- `validate_and_truncate_physical_indices()`：移除超出文档长度的页码。

这对金融 PDF 的脏数据有一定鲁棒性，但仍需项目侧抽检。正式赛题建议对索引结果做离线质量检查，例如：节点页码是否单调、页码范围是否为空、标题是否大量缺失、页文本是否乱码。

### 5.4 大节点递归拆分

`process_large_node_recursively()` 会检查节点页数和 Token 数。如果某个节点太大，并且超过配置 `max_page_num_each_node` 和 `max_token_num_each_node`，它会对该节点页段再次运行树抽取，生成更细粒度子节点。

默认配置是：

```yaml
toc_check_page_num: 20
max_page_num_each_node: 10
max_token_num_each_node: 20000
if_add_node_id: "yes"
if_add_node_summary: "yes"
if_add_doc_description: "no"
if_add_node_text: "no"
```

赛题可调建议：

- 法规/保险条款：`max_page_num_each_node` 可设为 3-6，便于条款级定位。
- 年报/研报：可设为 5-10，保留 MD&A、财务指标、业务章节等自然段落。
- 金融合同/募集说明书：可设为 4-8，重点保留“发行条款”“担保”“评级”“兑付”“募集资金用途”等章节。

## 6. Markdown 索引构建源码流程

`pageindex/page_index_md.py` 的流程比 PDF 简单：

```text
md_to_tree(md_path)
  -> extract_nodes_from_markdown()             # 提取 # 标题与行号
  -> extract_node_text_content()               # 每个标题到下个标题之间作为正文
  -> update_node_list_with_text_token_count()  # 可选，计算含子节点总 token
  -> tree_thinning_for_index()                 # 可选，过细小节点合并
  -> build_tree_from_nodes()                   # 用标题 level 构树
  -> write_node_id()                           # 可选
  -> generate_summaries_for_structure_md()     # 可选
  -> generate_doc_description()                # 可选
  -> format_structure()
```

Markdown 模式适合赛题的原因：

- 赛题允许在预处理阶段使用 OCR、版面分析、表格恢复、阅读顺序还原、PDF 转结构化文本等工具。
- 金融文档的标题、条款编号、表格标题很关键，Markdown 能比普通 PDF 文本更稳定地保留层级。
- `md_to_tree()` 不需要 LLM 去“发现”标题层级，只要 Markdown 标题规范，树构建是确定性的，成本低。

注意：如果 Markdown 是从 PDF/HTML 自动转换而来，但标题层级不可靠，`md_to_tree()` 的效果会下降。项目侧应在预处理阶段规范标题，例如把“第一章”“第十七条”“§ 3.2”“Item 7”等统一转成 Markdown 标题。

## 7. 检索工具源码解读

`pageindex/retrieve.py` 暴露三类工具函数，正好对应 Agentic RAG 的最小工具集。

### 7.1 `get_document(documents, doc_id)`

返回文档元信息，包括 `doc_id`、`doc_name`、`doc_description`、`type`、`status`、`page_count` 或 `line_count`。赛题中可用于确认文档是否索引完成，以及做日志记录。

### 7.2 `get_document_structure(documents, doc_id)`

返回树结构，但通过 `remove_fields(..., fields=['text'])` 移除正文。该设计非常适合赛题：先让 Qwen 看标题、页码、可选摘要，选择相关节点；不要一开始就把正文全部放进上下文。

### 7.3 `get_page_content(documents, doc_id, pages)`

按页或行读取原文，支持：

```text
"5-7"    # 连续页
"3,8"    # 离散页
"12"     # 单页
```

PDF 模式优先读取 `documents[doc_id]['pages']` 缓存，若没有缓存则回退到 `PyPDF2.PdfReader` 实时读取。Markdown 模式把 `pages` 理解为标题行号范围，会返回落在行号范围内的节点正文。

赛题中建议把该函数再包一层，加入：

- 页码范围合并。
- 最大页数限制。
- 前后页扩展，例如表格跨页时扩展 `±1` 页。
- OCR 文本优先读取，而不是 PyPDF2 原生抽取。
- 缓存命中统计。

## 8. 工具函数源码解读

`pageindex/utils.py` 中最值得复用的函数如下：

| 函数 | 用途 | 赛题用法 |
| --- | --- | --- |
| `count_tokens(text, model)` | 用 LiteLLM 估算 Token | 检索上下文预算控制。 |
| `extract_json(content)` | 从模型输出提取 JSON | 树检索、证据抽取、选项判断的 JSON 后处理可复用，但建议增强容错。 |
| `write_node_id(data)` | 为树节点写入递增 ID | 对自建树或预处理树补 ID。 |
| `structure_to_list(structure)` | 树转扁平节点列表 | 建节点索引、关键词匹配、候选 node 召回。 |
| `get_leaf_nodes(structure)` | 获取叶子节点 | 适合只在叶子章节中检索证据。 |
| `remove_fields(data, fields)` | 去除正文等字段 | 控制 Prompt 大小。 |
| `create_node_mapping(tree)` | `node_id -> node` 映射 | 从 Qwen 返回的 node_id 快速取页码范围。 |
| `get_page_tokens()` | PDF 按页抽文本并计 Token | 离线质量检查与页级缓存。 |
| `add_node_text()` | 给节点挂原文 | 只在离线构建阶段或调试时使用，正式检索不要把全部 text 传给模型。 |
| `generate_node_summary()` | 生成节点摘要 | 若参与正式检索，必须使用 Qwen 并计入成本或明确作为离线合规产物。 |

## 9. 高层客户端 `PageIndexClient` 解读

`PageIndexClient` 的主流程是：

```text
client = PageIndexClient(api_key=None, model=None, retrieve_model=None, workspace=None)
doc_id = client.index(file_path, mode="auto")
client.get_document(doc_id)
client.get_document_structure(doc_id)
client.get_page_content(doc_id, pages="5-7")
```

优点：

- 用法简单。
- 自动区分 PDF / Markdown。
- workspace 可持久化索引。
- 提供统一读取工具。

不足：

- `index()` 生成随机 UUID，不等于赛题 `doc_id`。
- `index()` 对 PDF 和 Markdown 都默认保留正文并生成摘要/文档描述，不适合严格控成本和控合规的赛题场景。
- 使用 `OPENAI_API_KEY` 作为默认兼容环境变量，需改为 DashScope / Qwen 时必须显式配置模型和环境变量。
- 未提供候选文档检索、题目解析、证据抽取、答案格式化、Token 统计。

因此建议：赛题生产代码不直接调用 `PageIndexClient.index()`；可以借鉴其 `documents` 字典格式和三个读取接口，在项目侧实现 `FinancePageIndexStore`。

## 10. Agent 示例源码解读

`examples/agentic_vectorless_rag_demo.py` 使用 OpenAI Agents SDK，定义了三个工具：

```text
get_document()
get_document_structure()
get_page_content(pages)
```

其系统提示词要求：

- 先调用 `get_document()` 确认页数。
- 再调用 `get_document_structure()` 找相关页段。
- 用紧凑页段调用 `get_page_content()`。
- 不要抓取全文。
- 仅基于工具输出回答。

这套思想适合赛题，但不建议直接使用 OpenAI Agents SDK。赛题应改成 Qwen API + 自定义工具循环或确定性流水线，原因是：

1. 赛题推理问答阶段必须使用 Qwen 系列模型。
2. 自定义流水线更容易做 Token 统计、缓存和答案约束。
3. 多选题需要逐选项判断，比通用聊天式 Agent 更适合结构化流程。

## 11. PageIndex 与赛题约束的合规边界

赛题允许在文档解析阶段使用 OCR、版面分析、表格恢复、阅读顺序还原、PDF 转结构化文本等工具，也允许这些预处理工具包含非 Qwen 模型。但正式检索、段落定位、记忆压缩、证据判断和答案生成阶段必须使用 Qwen 系列模型。

结合 PageIndex，需要遵循以下原则：

1. 如果 PageIndex 仅用来生成结构字段，例如标题、页码、层级、节点 ID，且这些字段属于文档结构解析结果，风险较低。
2. 如果 PageIndex 用非 Qwen 模型生成 `summary`、`doc_description`，并在正式答题中用这些语义字段做文档选择、节点选择或证据判断，则风险较高，应避免。
3. 若需要使用节点摘要参与检索，建议显式把 PageIndex 模型设置为 Qwen，并把相关调用记录纳入实验日志与必要的 Token 统计。
4. 如果希望把 PageIndex 完全作为预处理工具，并降低合规争议，建议关闭 `if_add_node_summary`、`if_add_doc_description`，树检索只使用标题、页码、条款号、章节号和项目侧关键词索引。

## 12. 不修改源码的接入方式

由于 PageIndex 仓库没有标准 `setup.py` 或 `pyproject.toml`，作为 submodule 使用时，推荐项目侧设置 `PYTHONPATH` 或在适配器入口插入路径。

项目结构示例：

```text
competition_project/
├── third_party/
│   └── PageIndex/                 # git submodule
├── agent/
│   ├── pageindex_adapter.py        # 项目侧封装，不改 submodule
│   ├── retriever.py
│   ├── qwen_client.py
│   └── answerer.py
├── processed_data/
│   ├── pageindex/
│   ├── pages/
│   └── doc_catalog.jsonl
└── scripts/
    ├── build_pageindex.py
    └── run_answer.py
```

环境变量方式：

```bash
export PYTHONPATH="$PWD/third_party/PageIndex:$PYTHONPATH"
```

代码内方式：

```python
from pathlib import Path
import sys

PAGEINDEX_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "PageIndex"
sys.path.insert(0, str(PAGEINDEX_ROOT))

from pageindex.page_index import page_index
from pageindex.page_index_md import md_to_tree
from pageindex.retrieve import get_document_structure, get_page_content
from pageindex.utils import remove_fields, structure_to_list, create_node_mapping
```

## 13. 推荐复用方式与避坑清单

### 13.1 推荐复用

- 离线索引：调用 `page_index()` 或 `md_to_tree()` 生成树。
- 在线检索：读取保存好的树 JSON，不重复跑 PageIndex 索引。
- 树检索：将 `structure` 去掉 `text` 后交给 Qwen 选择 node_id。
- 内容读取：根据 node 的 `start_index/end_index` 调用页级读取。
- 证据输出：保存 `doc_id`、`node_id`、`title`、`pages`、`quote`、`reasoning`。

### 13.2 谨慎使用

- `PageIndexClient.index()`：默认摘要和正文配置不适合控成本。
- `doc_description`：若用于 B 榜候选文档召回，应确保由 Qwen 生成或改为人工/元数据描述。
- `summary`：可提升树检索命中率，但会增加预处理成本和合规要求。
- PyPDF2 抽取：对扫描件、表格、双栏研报可能质量不足；建议优先使用高质量预处理文本。

### 13.3 不建议使用

- 非 Qwen embedding / rerank / semantic search 参与正式 B 榜候选检索。
- 非 Qwen 摘要、FAQ、结论提炼直接参与正式答题。
- 让 Agent 自由多轮工具调用且不设页数限制、调用次数限制和答案格式校验。

## 14. 面向赛题的源码级风险点

| 风险点 | 源码表现 | 影响 | 规避方式 |
| --- | --- | --- | --- |
| 默认模型不是 Qwen | `config.yaml` 默认模型是 OpenAI 系列 | 不符合推理阶段限制 | 显式传 `model="dashscope/qwen3.6-plus"` 或关闭语义摘要。 |
| `PageIndexClient.index()` 默认生成摘要和正文 | PDF/MD 调用均硬编码 summary/text/doc_description | 成本高，可能违规 | 直接调用底层 `page_index()` / `md_to_tree()`。 |
| PyPDF2 文本抽取质量不稳定 | `get_page_tokens()` 默认 PyPDF2 | 表格、扫描件、双栏错序 | 用 MinerU/OCR/版面分析生成 Markdown，再走 `md_to_tree()`。 |
| 树结构只有章节层级，不是候选文档检索系统 | 开源源码主要单文档检索 | B 榜 doc_id 缺失时不足 | 项目侧构建 doc catalog、元数据索引、关键词索引。 |
| LLM 调用没有暴露 usage | `llm_completion()` 丢弃 response usage | Token 统计困难 | 项目侧统一 Qwen 客户端；必要时 monkeypatch PageIndex LLM 函数。 |
| `extract_json()` 容错有限 | JSON 修复只处理少数情况 | Qwen 输出异常会导致空结果 | 项目侧增加 schema 校验、重试和保守回退。 |

## 15. 与金融文档类型的匹配分析

### 15.1 保险条款

适合用 PageIndex 定位：保险责任、责任免除、现金价值、退保、领取、身故保险金、等待期、犹豫期等章节。

建议在预处理中把“第 X 条”“保险责任”“责任免除”等转成标题节点。检索时按选项逐个定位公式与条件，避免只看摘要。

### 15.2 监管法规

适合用 PageIndex 定位：条、款、项、施行日期、报告期限、处罚条款、适用范围、例外条件。

建议以条文编号作为标题节点，证据输出必须保留法条原文和条文号。多选题逐项判断时，最好每个选项至少定位一个支持或反驳条款。

### 15.3 金融合同 / 债券募集说明书

适合定位：发行规模、期限、利率、评级、担保、兑付、募集资金用途、回售/赎回、违约责任。

建议对数值字段做结构化抽取辅助，但最终证据仍回到 PageIndex 页段原文。

### 15.4 财务报表

适合定位：年度经营指标、利润表、现金流量表、研发投入、分红政策、管理层讨论、风险因素。

注意表格跨页和 OCR 表格结构问题。PageIndex 负责定位章节，具体表格指标建议配合表格解析结果或页级文本后处理。

### 15.5 行业研报

适合定位：投资逻辑、行业趋势、公司比较、关键假设、风险提示、图表附近解释。

研报常见双栏、图表、脚注，PyPDF2 质量可能低。建议优先用版面分析转 Markdown，并保留图表标题和附近说明。

## 16. 建议的 PageIndex 索引产物格式

为了适配赛题，建议不要直接使用 PageIndex 的 UUID workspace，而是保存为项目自定义格式：

```json
{
  "doc_id": "strict_csrc_035",
  "domain": "regulatory",
  "title": "上市公司章程指引",
  "source_path": "data/raw/strict_csrc_035.pdf",
  "parse_path": "processed_data/markdown/strict_csrc_035.md",
  "index_type": "pageindex_md",
  "pageindex_model": "dashscope/qwen3.6-plus or none",
  "has_summary": false,
  "structure": [],
  "pages": [
    {"page": 1, "content": "..."}
  ],
  "quality": {
    "page_count": 120,
    "node_count": 86,
    "empty_page_ratio": 0.01,
    "build_status": "ok"
  }
}
```

这样做的好处是：

- 保留赛题原始 `doc_id`。
- 支持 A 榜直接按题目 `doc_ids` 找索引。
- 支持 B 榜候选召回后快速进入 PageIndex 树检索。
- 便于 `evidence.json` 引用原始文档 ID。
- 便于代码审核复现。

## 17. 最小可复用接口设计

项目侧可抽象成以下接口：

```python
class FinancePageIndexStore:
    def load_doc(self, doc_id: str) -> dict: ...
    def get_structure(self, doc_id: str, with_summary: bool = False) -> list[dict]: ...
    def flatten_nodes(self, doc_id: str) -> list[dict]: ...
    def get_node(self, doc_id: str, node_id: str) -> dict: ...
    def get_pages(self, doc_id: str, page_range: str) -> list[dict]: ...
    def node_to_page_range(self, doc_id: str, node_id: str, expand: int = 0) -> str: ...
```

这个接口将 PageIndex 隔离为“索引与读取后端”，上层金融 Agent 不直接依赖 PageIndex 内部实现，后续可以替换为自建树、章节索引或混合检索。

## 18. 总结

PageIndex 的源码实现核心是：用 LLM 与文档结构信号把长 PDF/Markdown 转成树，再让 Agent 基于树做可解释检索。对金融 Agent 问答赛题，它最适合承担“候选文档内部的结构化定位和按需取页”角色。

生产集成时的关键原则是：不要改 PageIndex 源码；不要直接用默认高层入口；不要让非 Qwen 语义摘要参与正式答题；不要把全文或全部节点正文交给模型。应在项目侧写薄封装，控制模型、字段、缓存、页数、Token 统计和答案格式，从而把 PageIndex 的树索引能力转化为赛题所需的高效、可追溯金融问答能力。
