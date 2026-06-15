## PageIndex 中 Agent（LLM）如何使用结构化索引

PageIndex 的核心思路是：**用 LLM 构建层次化树形索引，再用 LLM Agent 基于该索引进行推理式检索**，替代传统的向量相似度搜索 + 分块方案。整个过程分为两个阶段：

---

### 一、索引构建阶段：LLM 全程参与构建结构化索引

LLM 在构建索引时承担了以下关键角色（均在 [page_index.py](file:///Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/open_projects/PageIndex/pageindex/page_index.py) 中）：

#### 1. 目录检测与提取
- **`toc_detector_single_page`**（L104）：LLM 逐页判断某页是否包含目录（排除摘要、符号表等干扰项）
- **`extract_toc_content`**（L160）：LLM 从原始文本中提取完整目录内容，支持续写（防止输出截断）
- **`detect_page_index`**（L202）：LLM 判断目录中是否包含页码信息

#### 2. 目录结构化转换
- **`toc_transformer`**（L273）：LLM 将原始目录文本转换为 JSON 树形结构，包含 `structure`（如 "1.2.3"）、`title`、`page` 字段
- **`check_if_toc_transformation_is_complete`**（L143）：LLM 自检转换是否完整，不完整则自动续写

#### 3. 页码到物理页的映射
根据目录是否包含页码，分三种模式处理：

| 模式 | 函数 | LLM 作用 |
|------|------|----------|
| 目录有页码 | `process_toc_with_page_numbers`（L622） | `toc_index_extractor`（L243）用 LLM 将章节标题映射到 `<physical_index_X>` 标签 |
| 目录无页码 | `process_toc_no_page_numbers`（L597） | `add_page_number_to_toc`（L461）用 LLM 逐段匹配章节起始页 |
| 无目录 | `process_no_toc`（L576） | `generate_toc_init`（L542）/ `generate_toc_continue`（L507）用 LLM 直接从正文生成树形结构 |

#### 4. 验证与纠错
- **`verify_toc`**（L900）：LLM 抽样检查章节标题是否确实出现在映射的页面中（`check_title_appearance`）
- **`fix_incorrect_toc_with_retries`**（L878）：对验证不通过的条目，LLM 在前后正确条目限定的页面范围内重新定位（`single_toc_item_index_fixer`），最多重试 3 次
- 若准确率仍低于 60%，自动降级到更基础的模式重试（L992-997）

#### 5. 大节点递归处理
- **`process_large_node_recursively`**（L1000）：对于页数/Token 数超过阈值的节点，递归调用 `meta_processor` 在子范围内重新构建子索引

#### 6. 摘要与描述生成
- **`generate_node_summary`**（L578）：LLM 为每个节点生成内容摘要
- **`generate_doc_description`**（L622）：LLM 为整个文档生成一句话描述

---

### 二、检索阶段：Agent 基于结构化索引进行推理式检索

在 [agentic_vectorless_rag_demo.py](file:///Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/open_projects/PageIndex/examples/agentic_vectorless_rag_demo.py) 中，Agent 通过三个工具函数使用结构化索引：

```python
# Agent 的系统提示词（L44-52）
AGENT_SYSTEM_PROMPT = """
You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page/line count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
"""
```

#### 三个工具的分工：

| 工具 | 对应函数 | 作用 |
|------|----------|------|
| `get_document()` | [retrieve.py L81](file:///Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/open_projects/PageIndex/pageindex/retrieve.py#L81) | 获取文档元数据（名称、描述、页数/行数、状态） |
| `get_document_structure()` | [retrieve.py L100](file:///Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/open_projects/PageIndex/pageindex/retrieve.py#L100) | 获取**去除了 text 字段**的树形结构索引（节省 Token） |
| `get_page_content()` | [retrieve.py L110](file:///Users/lzc/TNTprojectZ/AprojectZ/AFAC2026/open_projects/PageIndex/pageindex/retrieve.py#L110) | 按页码范围获取原文内容 |

#### Agent 的推理流程：

```
用户提问
  → Agent 调用 get_document() 确认文档状态和规模
  → Agent 调用 get_document_structure() 获取树形结构索引
  → Agent 阅读结构中的 title + summary，推理出相关章节
  → Agent 调用 get_page_content(pages="X-Y") 精确获取相关页面原文
  → Agent 基于原文生成回答
```

**关键设计**：结构索引中的 `summary` 字段（由 LLM 在构建阶段生成）让 Agent 无需读取全文就能判断哪些章节与问题相关，然后通过 `start_index`/`end_index` 精确定位到具体页面范围，实现"按需取页"的检索策略。

---

### 总结

PageIndex 中 LLM 扮演了**双重角色**：

1. **索引构建者**：LLM 负责从 PDF 中提取目录、映射页码、生成树形结构、验证纠错、生成摘要——整个索引本身就是 LLM 的产物
2. **索引消费者**：Agent 在检索时阅读树形结构（含摘要），通过推理判断相关章节，再精确获取原文——这是"agentic reasoning over structured index"而非传统的向量相似度匹配