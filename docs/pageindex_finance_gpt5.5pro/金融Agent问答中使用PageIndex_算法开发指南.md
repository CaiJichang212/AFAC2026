# 在“金融长文本 Agent 问答”任务中使用 PageIndex：算法开发指南

本文档给出一个不修改 PageIndex 源码、仅把 PageIndex 作为第三方子模块使用的工程方案。读者对象是赛题算法开发人员，重点关注：如何构建索引、如何在 A/B 榜流程中使用 PageIndex、如何控制 Qwen 调用、如何做证据检索、记忆压缩、答案规范化和 Token 统计。

## 1. 总体定位

PageIndex 在本任务中的定位是“文档内部结构化检索后端”，不是完整的金融问答系统。

完整系统应拆成四层：

```text
原始金融文档
  -> 文档解析层：OCR / 版面分析 / 表格恢复 / Markdown 化 / 页缓存
  -> PageIndex 结构索引层：PDF 或 Markdown -> 章节树 / node_id / 页码或行号
  -> 金融检索与记忆层：候选文档召回、树检索、页级证据读取、证据压缩
  -> 答案推理层：逐选项判断、数值计算、答案校验、Token 统计、answer.csv/evidence.json
```

PageIndex 负责第二层，并给第三层提供 `structure` 与 `page_content`。候选文档召回、证据判断和答案生成必须由项目侧实现。

## 2. 推荐系统架构

项目根目录为 `AFAC2026/`，PageIndex 作为 git submodule 放在 `open_projects/PageIndex/`，赛题数据在 `data/public_dataset_upload/`。

```text
AFAC2026/
├── open_projects/
│   └── PageIndex/                         # git submodule，不修改
├── data/
│   └── public_dataset_upload/
│       ├── raw/                           # 原始 PDF/HTML 文档
│       │   ├── financial_contracts/       # 金融合同（债券募集说明书）
│       │   ├── financial_reports/         # 上市公司年度报告
│       │   ├── insurance/                 # 保险产品条款
│       │   └── regulatory/               # 监管法规
│       │       ├── html/                  # 法规 HTML
│       │       └── attachments/           # 法规附件 PDF
│       └── questions/
│           └── group_a/
│               ├── financial_contracts_questions.json
│               ├── financial_reports_questions.json
│               ├── insurance_questions.json
│               ├── regulatory_questions.json
│               └── research_questions.json
├── processed_data/
│   ├── markdown/                          # PDF 解析后的 Markdown，可选
│   ├── pages/                             # 页级文本缓存
│   ├── pageindex/                         # PageIndex 树索引
│   ├── catalog/doc_catalog.jsonl          # B榜候选文档检索用元数据
│   └── quality/index_quality.jsonl
├── agent/
│   ├── pageindex_adapter.py               # 封装 PageIndex
│   ├── qwen_client.py                     # 统一 Qwen API 与 Token 统计
│   ├── question_parser.py                 # 题目解析
│   ├── doc_retriever.py                   # A/B 榜候选文档检索
│   ├── tree_retriever.py                  # PageIndex 树检索
│   ├── evidence_extractor.py              # 页级证据抽取与压缩
│   ├── answerer.py                        # 逐选项判断与答案规范化
│   └── token_meter.py
├── scripts/
│   ├── build_pageindex.py                 # 离线构建索引
│   ├── run_eval.py                        # 跑题生成结果
│   └── export_submission.py
├── docs/                                  # 赛题分析文档
│   ├── finance_agent_gpt5.5thinking/
│   └── pageindex_finance_gpt5.5pro/
├── outputs/
│   ├── answer.csv
│   ├── evidence.json
│   └── logs/
└── README.md
```

## 3. 环境接入方式

PageIndex 作为 submodule 放在 `open_projects/PageIndex/`，不需要安装成包。推荐在运行脚本中设置 `PYTHONPATH`：

```bash
git submodule update --init --recursive
pip install -r open_projects/PageIndex/requirements.txt
export PYTHONPATH="$PWD/open_projects/PageIndex:$PYTHONPATH"
```

如果通过 DashScope 调用 Qwen，并让 PageIndex 内部的 LiteLLM 直接调用 Qwen，可设置：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

在 PageIndex 调用中显式传入模型名，例如：

```python
MODEL = "dashscope/qwen3.6-plus"
```

实际模型名应以赛题平台、百炼或魔搭社区提供的可用模型 ID 为准。不要使用 PageIndex 默认 `config.yaml` 中的 OpenAI 系列模型作为正式推理或语义检索模型。

## 4. 离线索引构建策略

### 4.1 两条可选路线

路线 A：直接 PDF -> PageIndex。

```text
PDF -> PyPDF2/PyMuPDF 抽页文本 -> PageIndex LLM 目录/树构建 -> structure.json
```

优点是接入简单；缺点是金融 PDF 的表格、扫描件、双栏研报、复杂页眉页脚可能抽取不稳。

路线 B：PDF -> 高质量 Markdown -> PageIndex Markdown 树。

```text
PDF -> OCR/版面分析/表格恢复/阅读顺序还原 -> Markdown -> md_to_tree() -> structure.json
```

优点是结构更可控、成本更低、可复现性更强。若预处理能把条款编号、章节标题和表格标题转成 Markdown 标题，路线 B 更适合本赛题。

### 4.2 合规建议

为了避免非 Qwen 语义产物参与正式答题，推荐默认关闭 PageIndex 摘要和文档描述：

```python
if_add_node_summary="no"
if_add_doc_description="no"
if_add_node_text="no"
```

如果确实要用 `summary` 提升树检索效果，应使用 Qwen 生成，并记录调用日志。对于非 Qwen 预处理工具，建议只使用其 OCR、版面、标题、表格恢复能力，不把其生成的语义摘要、FAQ、结论提炼、向量召回结果用于正式答题。

### 4.3 PDF 索引构建示例

项目侧新建 `agent/pageindex_adapter.py`，不要改 PageIndex 源码：

```python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PAGEINDEX_ROOT = Path(__file__).resolve().parents[1] / "open_projects" / "PageIndex"
if str(PAGEINDEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PAGEINDEX_ROOT))

from pageindex.page_index import page_index


def build_pdf_pageindex(
    doc_id: str,
    pdf_path: str | Path,
    output_dir: str | Path,
    model: str = "dashscope/qwen3.6-plus",
    add_summary: bool = False,
) -> Path:
    """构建单个 PDF 的 PageIndex 树索引。"""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = page_index(
        doc=str(pdf_path),
        model=model,
        toc_check_page_num=20,
        max_page_num_each_node=6,
        max_token_num_each_node=12000,
        if_add_node_id="yes",
        if_add_node_summary="yes" if add_summary else "no",
        if_add_doc_description="no",
        if_add_node_text="no",
    )

    payload = {
        "doc_id": doc_id,
        "source_path": str(pdf_path),
        "index_type": "pageindex_pdf",
        "pageindex_model": model if add_summary else model,
        "has_summary": add_summary,
        "doc_name": result.get("doc_name", pdf_path.name),
        "structure": result["structure"],
    }

    out_path = output_dir / f"{doc_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
```

注意：即使关闭摘要，`page_index()` 在 PDF 模式下仍会调用模型来识别/生成文档树。因此如果该树构建发生在正式答题阶段，应使用 Qwen 并记录 Token。更推荐把索引构建放到题目无关的离线预处理阶段。

### 4.4 Markdown 索引构建示例

如果已有结构化 Markdown，优先使用 `md_to_tree()`：

```python
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

PAGEINDEX_ROOT = Path(__file__).resolve().parents[1] / "open_projects" / "PageIndex"
if str(PAGEINDEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PAGEINDEX_ROOT))

from pageindex.page_index_md import md_to_tree


def build_md_pageindex(
    doc_id: str,
    md_path: str | Path,
    output_dir: str | Path,
    model: str = "dashscope/qwen3.6-plus",
    add_summary: bool = False,
) -> Path:
    """构建单个 Markdown 的 PageIndex 树索引。"""
    md_path = Path(md_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(md_to_tree(
        md_path=str(md_path),
        if_thinning=False,
        min_token_threshold=5000,
        if_add_node_summary="yes" if add_summary else "no",
        summary_token_threshold=200,
        model=model,
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
    ))

    payload = {
        "doc_id": doc_id,
        "source_path": str(md_path),
        "index_type": "pageindex_md",
        "pageindex_model": model if add_summary else None,
        "has_summary": add_summary,
        "doc_name": result.get("doc_name", md_path.stem),
        "line_count": result.get("line_count"),
        "structure": result["structure"],
    }

    out_path = output_dir / f"{doc_id}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
```

Markdown 模式在不生成摘要时基本不需要 LLM 调用，适合大规模离线索引。

## 5. 索引质量检查

离线构建后，应生成 `processed_data/quality/index_quality.jsonl`。建议至少检查：

```python
from __future__ import annotations

from typing import Any


def flatten_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    def walk(items):
        for n in items:
            out.append(n)
            if n.get("nodes"):
                walk(n["nodes"])
    walk(nodes)
    return out


def check_structure_quality(doc_id: str, structure: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = flatten_nodes(structure)
    bad_ranges = []
    titles = []
    for n in nodes:
        titles.append(n.get("title", ""))
        s, e = n.get("start_index"), n.get("end_index")
        if s is not None and e is not None and s > e:
            bad_ranges.append({"node_id": n.get("node_id"), "start": s, "end": e})

    return {
        "doc_id": doc_id,
        "node_count": len(nodes),
        "empty_title_count": sum(1 for t in titles if not str(t).strip()),
        "bad_range_count": len(bad_ranges),
        "bad_ranges": bad_ranges[:10],
    }
```

质量检查用于决定哪些文档需要回退到更强 OCR、人工规则标题抽取或直接页级关键词检索。

## 6. 索引存储与读取封装

建议在项目侧实现统一 Store，而不是使用 PageIndex 的随机 UUID workspace。

```python
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class FinancePageIndexStore:
    def __init__(self, index_dir: str | Path, page_dir: str | Path | None = None):
        self.index_dir = Path(index_dir)
        self.page_dir = Path(page_dir) if page_dir else None

    @lru_cache(maxsize=512)
    def load_doc(self, doc_id: str) -> dict[str, Any]:
        path = self.index_dir / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"PageIndex not found for doc_id={doc_id}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_structure(self, doc_id: str) -> list[dict[str, Any]]:
        return self.load_doc(doc_id)["structure"]

    def flatten_nodes(self, doc_id: str) -> list[dict[str, Any]]:
        result = []
        def walk(nodes):
            for node in nodes:
                result.append({k: v for k, v in node.items() if k != "nodes"})
                if node.get("nodes"):
                    walk(node["nodes"])
        walk(self.get_structure(doc_id))
        return result

    def get_node(self, doc_id: str, node_id: str) -> dict[str, Any]:
        for node in self.flatten_nodes(doc_id):
            if node.get("node_id") == node_id:
                return node
        raise KeyError(f"node_id={node_id} not found in doc_id={doc_id}")

    def node_to_page_range(self, doc_id: str, node_id: str, expand: int = 0) -> str:
        node = self.get_node(doc_id, node_id)
        start = node.get("start_index")
        end = node.get("end_index")
        if start is None or end is None:
            raise ValueError(f"node has no page range: doc_id={doc_id}, node_id={node_id}")
        start = max(1, int(start) - expand)
        end = max(start, int(end) + expand)
        return f"{start}-{end}" if start != end else str(start)
```

页级文本建议由项目预处理保存到 `processed_data/pages/{doc_id}.jsonl`，不要依赖 PyPDF2 实时读取。读取接口示例：

```python
    @lru_cache(maxsize=4096)
    def get_page(self, doc_id: str, page: int) -> str:
        if self.page_dir is None:
            raise RuntimeError("page_dir is not configured")
        path = self.page_dir / f"{doc_id}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if int(item["page"]) == page:
                return item["content"]
        return ""

    def get_pages_by_range(self, doc_id: str, page_range: str, max_pages: int = 8) -> list[dict[str, Any]]:
        pages = parse_page_range(page_range)
        pages = pages[:max_pages]
        return [{"page": p, "content": self.get_page(doc_id, p)} for p in pages]


def parse_page_range(page_range: str) -> list[int]:
    pages = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(pages)
```

## 7. A 榜与 B 榜流程差异

### 7.1 A 榜

A 榜题目提供 `doc_ids`。流程应直接跳过候选文档召回：

```text
question.doc_ids
  -> 读取这些 doc_id 的 PageIndex structure
  -> 树检索相关 node_id
  -> 读取页级内容
  -> 证据抽取/压缩
  -> 逐选项判断
  -> 输出答案
```

### 7.2 B 榜

B 榜没有 `doc_ids`，需要先候选文档检索：

```text
question.domain + question/options/type
  -> domain 过滤
  -> 元数据/标题/文件名/章节标题关键词召回
  -> Qwen 候选文档选择或重排
  -> 对 top-k 文档运行 PageIndex 树检索
  -> 证据抽取/答案判断
```

B 榜候选召回建议优先级：

1. `domain` 过滤。题目字段若提供领域，先限定在对应领域。
2. 元数据规则。公司名、产品名、法规名、年份、债券简称、行业关键词。
3. 关键词/SQLite FTS/BM25。只基于原文、标题、章节名、人工元数据；不要使用非 Qwen embedding。
4. Qwen 文档选择。输入紧凑 doc catalog，输出候选 doc_ids。
5. PageIndex 树内证据验证。候选文档不是答案，必须继续定位证据页。

## 8. 文档候选召回设计

### 8.1 文档目录 catalog

为 B 榜准备 `doc_catalog.jsonl`。根据赛题实际数据，doc_id 与原始文件名对应（不含扩展名），领域分为五类：`financial_contracts`（金融合同）、`financial_reports`（财务报表）、`insurance`（保险）、`regulatory`（监管法规）、`research`（研报）：

```json
{"doc_id":"annual_byd_2025_report","domain":"financial_reports","title":"比亚迪股份有限公司2025年年度报告","company":"比亚迪","year":"2025","top_titles":["管理层讨论与分析","财务报告","研发投入"]}
{"doc_id":"text01","domain":"financial_contracts","title":"广东省广晟控股集团有限公司债券募集说明书","issuer":"广东省广晟控股集团有限公司","top_titles":["发行条款","发行人基本情况","中介机构"]}
{"doc_id":"1","domain":"insurance","title":"平安智盈金生产品条款","product":"平安智盈金生","top_titles":["保险责任","身故保险金","现金价值"]}
{"doc_id":"strict_v3_008_中国人民银行令〔2025〕第12号（金融机构客户受益所有人识别管理办法）","domain":"regulatory","title":"金融机构客户受益所有人识别管理办法","year":"2025","top_titles":["受益所有人识别","信息核对","报告义务"]}
```

`top_titles` 可以来自 PageIndex 的一级/二级标题，但不要包含非 Qwen 语义摘要。这样即使没有向量检索，也能用关键词和 Qwen 做候选选择。

### 8.2 Qwen 文档选择 Prompt

```text
你是金融长文本问答系统的候选文档选择器。
只根据给定的文档元数据、标题和章节标题选择可能包含答案的文档。
不要作答，不要猜测标准答案。

题目：{question}
选项：{options}
领域：{domain}
候选文档目录：{compact_doc_catalog}

输出 JSON：
{
  "doc_ids": ["..."],
  "reason": "简述选择依据，不超过80字"
}
要求：
1. 最多返回 {k} 个 doc_id。
2. 如果题目明显是跨年/跨产品/跨法规比较，可返回多个文档。
3. 不要返回目录中不存在的 doc_id。
```

候选文档选择本身属于正式检索阶段，必须使用 Qwen，并纳入 Token 统计。

## 9. PageIndex 树检索设计

树检索有两种实现方式。

### 9.1 小树：直接让 Qwen 看完整结构

适合节点数少于 100、结构不长的文档。

```python
import json


def compact_tree(structure: list[dict], keep_summary: bool = False) -> list[dict]:
    allowed = {"title", "node_id", "start_index", "end_index", "line_num"}
    if keep_summary:
        allowed |= {"summary", "prefix_summary"}

    def clean(node):
        out = {k: v for k, v in node.items() if k in allowed and v is not None}
        if node.get("nodes"):
            out["nodes"] = [clean(ch) for ch in node["nodes"]]
        return out

    return [clean(n) for n in structure]


def build_tree_search_prompt(question: dict, doc_id: str, tree: list[dict]) -> str:
    return f"""
你是金融长文本问答系统的树检索器。
任务：根据题干和选项，在给定文档树中找出最可能包含证据的节点。

文档ID：{doc_id}
题干：{question['question']}
题型：{question.get('answer_format')}
选项：{json.dumps(question.get('options', {}), ensure_ascii=False)}

文档树：
{json.dumps(tree, ensure_ascii=False)}

输出 JSON：
{{
  "nodes": [
    {{"node_id": "0001", "need_expand": true, "reason": "..."}}
  ]
}}
要求：
1. 返回 3-8 个节点，优先返回页码范围更小且标题最相关的节点。
2. 若题目是多文档/跨年比较，保留所有可能相关指标节点。
3. 不要作答，只选择节点。
4. 不要返回不存在的 node_id。
""".strip()
```

### 9.2 大树：规则预筛 + Qwen 选择

对节点很多的文档，不要把完整树传入模型。先在项目侧做关键词预筛，再让 Qwen 选择。

```python
import re
from collections import Counter


def extract_query_terms(question: dict) -> list[str]:
    text = question["question"] + "\n" + "\n".join(question.get("options", {}).values())
    # 可按领域扩展：金额、百分比、年份、条款号、指标名等
    terms = re.findall(r"[\u4e00-\u9fffA-Za-z0-9.%]+", text)
    terms = [t for t in terms if len(t) >= 2]
    return terms


def lexical_node_candidates(nodes: list[dict], terms: list[str], topn: int = 40) -> list[dict]:
    scored = []
    for n in nodes:
        title = str(n.get("title", ""))
        summary = str(n.get("summary", ""))
        hay = title + " " + summary
        score = sum(1 for t in terms if t in hay)
        # 金融常见强信号可加权
        if any(x in title for x in ["保险责任", "责任免除", "现金价值", "募集资金", "研发投入", "现金流量", "股东大会", "处罚", "评级"]):
            score += 2
        if score > 0:
            scored.append((score, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [n for _, n in scored[:topn]]
```

把 `topn` 候选节点以扁平列表传给 Qwen，比传完整树更省 Token。

## 10. 页段读取与证据窗口控制

从节点到页段时，建议：

- 默认读取节点页码范围。
- 如果节点只有 1-2 页，可扩展前后各 1 页。
- 如果节点超过 8 页，不直接读全；先按关键词在该页段内检索页，再读取命中页及邻页。
- 表格题优先扩展后一页，因为表头/表体常跨页。
- 法规题优先保持条文附近短窗口，避免把无关法条混入。

页段合并函数：

```python
def merge_page_ranges(ranges: list[tuple[int, int]], max_total_pages: int = 12) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        last = merged[-1]
        if s <= last[1] + 1:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])

    out = []
    total = 0
    for s, e in merged:
        length = e - s + 1
        if total + length > max_total_pages:
            remain = max_total_pages - total
            if remain > 0:
                out.append((s, s + remain - 1))
            break
        out.append((s, e))
        total += length
    return out
```

## 11. 证据抽取与动态记忆压缩

不要把页级原文直接累积到最终回答 Prompt。应把每次读取的页段压缩成“证据记忆”。证据记忆必须保留原文引文和定位信息。

建议结构：

```json
{
  "qid": "reg_a_014",
  "doc_id": "strict_csrc_035",
  "node_id": "0047",
  "title": "第四十七条 对外担保",
  "pages": "12-13",
  "option": "A",
  "evidence_type": "support",
  "quote": "第四十七条 公司下列对外担保行为，须经股东会审议通过：...",
  "normalized_fact": "为资产负债率超过70%的担保对象提供担保须经股东会审议。",
  "numbers": {"threshold": "70%", "given": "75%"},
  "confidence": "high"
}
```

证据抽取 Prompt：

```text
你是金融问答证据抽取器。
只基于给定原文，抽取能判断指定选项真假的证据。
不得使用常识，不得补充原文没有的信息。

题干：{question}
选项 {label}: {option_text}
文档ID：{doc_id}
页码：{pages}
原文：
{page_text}

输出 JSON：
{
  "option": "A",
  "evidence_type": "support/refute/unclear",
  "quote": "原文短引文，必须来自原文",
  "normalized_fact": "对证据的简短归纳",
  "numbers": {"字段": "值"},
  "reason": "为什么该证据支持/反驳/无法判断，不超过120字"
}
```

动态记忆压缩策略：

1. 每个选项维护独立证据池。
2. 证据去重：同一 `doc_id + pages + quote` 只保留一次。
3. 冲突保留：同一选项若同时有 support/refute，保留两类证据给最终判断器。
4. 数值题保留结构化数字，不只保留自然语言摘要。
5. 压缩只压缩“解释”，不删除定位、引文、数字和证据类型。

## 12. 逐选项判断与答案生成

最终判断不应让模型重新检索，而应基于证据记忆逐选项判断。

```text
你是金融长文本选择题判断器。
只根据给定证据判断选项。没有证据支持时不要猜测。
输出必须是 JSON。

题干：{question}
题型：{answer_format}
选项：{options}
证据：{evidence_memory}

判断规则：
1. 单选题只选择唯一正确选项。
2. 多选题选择所有正确选项；漏选、错选、多选都会错误。
3. 判断题按题目选项含义输出 A 或 B。
4. 如选项包含数值比较，必须根据证据中的数字或程序计算结果判断。
5. 不要输出解释到 answer 字段。

输出 JSON：
{
  "option_judgements": {
    "A": {"verdict": true, "reason": "...", "evidence_ids": [0]},
    "B": {"verdict": false, "reason": "...", "evidence_ids": [1]}
  },
  "answer": "AC"
}
```

答案后处理必须程序化：

```python
VALID = {"A", "B", "C", "D"}


def normalize_answer(raw_answer: str, answer_format: str) -> str:
    letters = [ch for ch in str(raw_answer).upper() if ch in VALID]
    if answer_format in {"mcq", "tf"}:
        return letters[0] if letters else "A"
    if answer_format == "multi":
        return "".join(sorted(set(letters)))
    return letters[0] if letters else "A"
```

## 13. 数值计算策略

金融题中常见金额、比例、增长率、现金流比较、净资产占比等。建议由程序计算，不完全依赖模型口算。

示例：

```python
from decimal import Decimal, ROUND_HALF_UP


def pct(numerator: str, denominator: str, digits: int = 2) -> Decimal:
    x = Decimal(numerator) / Decimal(denominator) * Decimal("100")
    q = Decimal("1." + "0" * digits)
    return x.quantize(q, rounding=ROUND_HALF_UP)

# 8亿元 / 120亿元 = 6.67%
print(pct("8", "120"))
```

证据抽取阶段让 Qwen 抽取数字和单位，项目侧统一归一化单位和计算，再把计算结果放回最终判断 Prompt。

## 14. Token 统计与缓存

### 14.1 统一 Qwen Client

所有正式检索、压缩、证据判断、答案生成和自检调用都应经过统一客户端，记录 usage。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)


class QwenClient:
    def __init__(self, model: str, low_level_client: Any):
        self.model = model
        self.client = low_level_client
        self.usage = TokenUsage()

    def chat_json(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.usage.add(usage.prompt_tokens, usage.completion_tokens)
        content = resp.choices[0].message.content
        return safe_json_loads(content)
```

`safe_json_loads()` 应实现代码块清理、尾逗号修复、schema 校验和重试。

### 14.2 PageIndex 内部调用的统计

PageIndex 源码中的 `llm_completion()` 和 `llm_acompletion()` 没有返回 usage。若正式流程中需要运行 PageIndex LLM 索引或摘要，并且这些调用需要计入统计，可以用运行时 monkeypatch，而不修改源码文件。

思路：

```python
import pageindex.page_index as pi_pdf
import pageindex.page_index_md as pi_md
import pageindex.utils as pi_utils

# 用项目侧函数替换模块中的 llm_completion / llm_acompletion。
# 注意：page_index.py 使用 `from .utils import *`，因此要同时 patch pi_pdf 里的名字。
pi_utils.llm_completion = tracked_llm_completion
pi_utils.llm_acompletion = tracked_llm_acompletion
pi_pdf.llm_completion = tracked_llm_completion
pi_pdf.llm_acompletion = tracked_llm_acompletion
pi_md.llm_completion = tracked_llm_completion
pi_md.llm_acompletion = tracked_llm_acompletion
```

更简单的工程建议是：PageIndex 索引构建作为题目无关的离线预处理；正式答题阶段只读取索引文件，不触发 PageIndex 内部 LLM 调用。

### 14.3 缓存策略

建议缓存：

- `doc_id -> structure`。
- `doc_id -> flattened nodes`。
- `doc_id + page -> page_text`。
- `qid + doc_id -> selected node_ids`。
- `qid + doc_id + pages + option -> evidence extraction result`。

禁止缓存：

- B 榜标准答案相关信息。
- 非 Qwen 生成的正式召回/重排/纠错结果。

## 15. 领域专项检索提示

### 15.1 保险条款

赛题保险数据为 16 份保险产品条款 PDF（`1.pdf` ~ `16.pdf`），涉及年金险、医疗险、重疾险等。doc_id 为数字编号。题目多为多产品比较、公式计算（身故保险金、退保金、赔付金额）。

查询扩展词：

```text
保险责任、责任免除、身故保险金、现金价值、账户价值、退保、犹豫期、等待期、领取、给付比例、已交保费、有效保险金额
```

检索策略：先定位条款章节，再读取公式所在页。多个产品比较题需要每个产品都提取同类字段，最后统一比较。

### 15.2 监管法规

赛题监管数据包含 200 份证监会法规 HTML 及其附件 PDF，doc_id 格式为 `strict_v3_XXX_法规名称`。题目多为跨法规比较、条文细节判断。

查询扩展词：

```text
适用范围、应当、不得、可以、报告、备案、审议、表决、处罚、期限、工作日、股东会、董事会、特别决议、普通决议
```

检索策略：条文编号优先。最终 evidence 必须保留条文号和短引文。不要用常识替代法条。

### 15.3 金融合同 / 债券

赛题金融合同数据为 14 份债券募集说明书摘要 PDF（`text01.pdf` ~ `text14.pdf`），doc_id 为 `text01` ~ `text14`。题目多为跨债券条款比较。

查询扩展词：

```text
发行规模、期限、票面利率、还本付息、担保、评级、募集资金用途、回售、赎回、违约、受托管理、偿债保障
```

检索策略：先定位“发行条款概要”或“募集说明书摘要”，再定位正文细则。冲突时以正文条款为准。

### 15.4 财务报表

赛题财报数据为 10 份上市公司年度报告 PDF，涵盖比亚迪、宁德时代、中国移动、招商银行、中国建筑、美的集团六家公司两年对比。doc_id 格式为 `annual_{公司英文名}_{年份}_report`。

查询扩展词：

```text
营业收入、归母净利润、扣非净利润、经营活动现金流量净额、研发投入、分红、资产负债率、毛利率、管理层讨论与分析
```

检索策略：页级定位后，尽量抽取表格附近原文。跨年比较要保持两份文档的同口径指标。

### 15.5 行业研报

赛题研报数据（questions 在 `research_questions.json` 中，doc_id 格式为 `pack2_textXX`）。题目多为市场规模预测、行业比较、数据核验。

查询扩展词：

```text
投资建议、行业趋势、市场规模、竞争格局、公司比较、风险提示、盈利预测、核心观点、摘要、结论
```

检索策略：研报标题与图表标题很重要。读取节点页时可扩展后页，避免图表说明跨页遗漏。

## 16. 端到端运行流程

### 16.1 离线阶段

```text
1. 读取赛题原始文档目录 data/public_dataset_upload/raw/ 下的所有 PDF/HTML。
2. 对每个 PDF 做解析，生成 pages 缓存和可选 Markdown。
3. 调用 PageIndex 生成结构树，doc_id 采用原始文件名（不含扩展名）。
4. 保存 processed_data/pageindex/{doc_id}.json。
5. 生成 doc_catalog.jsonl（基于文件名、领域、章节标题等元数据）。
6. 跑索引质量检查。
```

示例命令：

```bash
python scripts/build_pageindex.py \
  --raw-dir data/public_dataset_upload/raw \
  --markdown-dir processed_data/markdown \
  --page-dir processed_data/pages \
  --index-dir processed_data/pageindex \
  --model dashscope/qwen3.6-plus \
  --no-summary
```

### 16.2 在线答题阶段

```text
for question in questions:
    parse question and options
    if A split and doc_ids exists:
        candidate_docs = question.doc_ids
    else:
        candidate_docs = retrieve_candidate_docs(question)

    evidence_memory = []
    for doc_id in candidate_docs:
        structure = store.get_structure(doc_id)
        selected_nodes = retrieve_nodes_with_pageindex_tree(question, doc_id, structure)
        page_ranges = node_ids_to_page_ranges(selected_nodes)
        raw_pages = store.get_pages_by_ranges(doc_id, page_ranges)
        evidence_memory += extract_evidence_by_option(question, raw_pages)

    answer = judge_options_and_normalize(question, evidence_memory)
    write evidence.json item
    write answer.csv row with token usage delta
write summary row
```

## 17. `answer.csv` 与 `evidence.json` 输出

`answer.csv`：

```csv
qid,answer,prompt_tokens,completion_tokens,total_tokens
summary,,3627557,629,3628186
fin_a_001,AC,12000,300,12300
```

`evidence.json` 建议 JSONL，每题一行：

```json
{
  "qid": "fin_a_001",
  "answer": "AC",
  "candidate_docs": ["byd_2024_annual", "byd_2025_annual"],
  "selected_nodes": [
    {"doc_id": "byd_2024_annual", "node_id": "0012", "title": "主要会计数据和财务指标", "pages": "8-10"}
  ],
  "evidence_retrieval": [
    {
      "doc_id": "byd_2024_annual",
      "node_id": "0012",
      "pages": "8-10",
      "quote": "...",
      "reasoning": "..."
    }
  ],
  "usage": {"prompt_tokens": 12000, "completion_tokens": 300, "total_tokens": 12300}
}
```

## 18. 自检与回退策略

建议实现轻量自检，不要高成本多轮投票。

### 18.1 自检触发条件

- 多选题答案为空或为 ABCD。
- 某个选项没有任何证据。
- 支持/反驳证据冲突。
- 数值计算缺失或单位不一致。
- 候选文档数为 0。

### 18.2 回退顺序

```text
1. 扩大同节点页码窗口。
2. 增加同文档相邻节点。
3. 增加候选文档 top-k。
4. 使用领域关键词重新树检索。
5. 最后才进行更大上下文的 Qwen 判断。
```

### 18.3 严格预算

设置全局预算：

```python
MAX_DOCS_PER_QUESTION = 4
MAX_NODES_PER_DOC = 6
MAX_PAGES_PER_DOC = 10
MAX_EVIDENCE_PER_OPTION = 3
MAX_RETRY_PER_QUESTION = 1
```

这些阈值可按 A/B 榜和领域调整。

## 19. PageIndex 与其他检索的组合

推荐组合方式：

```text
B榜候选文档召回：元数据 + 关键词/FTS + Qwen 文档选择
文档内粗定位：PageIndex tree search
页内精定位：关键词窗口 + Qwen 证据抽取
最终判断：Qwen + 程序化计算/答案规范化
```

不推荐组合方式：

```text
非 Qwen embedding -> rerank -> 正式召回结果
非 Qwen 摘要/FAQ -> 直接判断答案
全文直接塞给 Qwen -> 输出字母
PageIndexClient.index 默认摘要/text -> 直接作为答题上下文
```

## 20. 关键 Prompt 模板汇总

### 20.1 树节点选择

```text
你是金融长文本问答系统的树检索器。
根据题目和选项，从文档树中选择最可能包含证据的节点。
只返回 node_id，不要回答题目。

题目：{question}
选项：{options}
文档树：{tree}

输出 JSON：
{"node_ids": ["0001", "0002"], "reason": "不超过80字"}
```

### 20.2 证据抽取

```text
你是金融证据抽取器。
只基于原文判断该页是否包含能支持或反驳选项的证据。

题目：{question}
选项：{label}. {option}
原文：{context}

输出 JSON：
{"evidence_type":"support/refute/unclear","quote":"...","fact":"...","numbers":{},"reason":"..."}
```

### 20.3 选项判断

```text
你是金融选择题判断器。
只根据证据判断每个选项真假，并输出合法答案字母。

题目：{question}
题型：{answer_format}
选项：{options}
证据：{evidence_memory}

输出 JSON：
{"judgements":{"A":true,"B":false,"C":true,"D":false},"answer":"AC"}
```

## 21. 开发迭代建议

第一阶段：只做 A 榜。

- 使用题目给定 `doc_ids`。
- 每题每文档完整树检索。
- 读取 top 3-5 节点页段。
- 逐选项证据抽取和答案判断。

第二阶段：优化 A 榜成本。

- 规则预筛节点。
- 缓存页级文本和树检索结果。
- 限制页数和证据数。
- 程序化计算常见数值题。

第三阶段：扩展 B 榜。

- 建 doc catalog。
- 做 domain + 元数据 + 关键词召回。
- Qwen 候选文档选择。
- 候选文档证据验证。

第四阶段：领域专项增强。

- 保险公式解析。
- 法规条文定位。
- 财报指标同口径比较。
- 研报图表/标题附近证据抽取。

## 22. 最小可行版本配置

建议从以下配置开始：

```yaml
pageindex:
  use_markdown_first: true
  add_summary: false
  max_page_num_each_node: 6
  max_token_num_each_node: 12000
retrieval:
  max_docs_a: 3
  max_docs_b: 4
  max_nodes_per_doc: 5
  max_pages_per_doc: 8
  expand_pages: 1
answer:
  max_evidence_per_option: 3
  self_check: true
  max_retry: 1
model:
  qwen: dashscope/qwen3.6-plus
```

## 23. 代码审核友好性

为了满足复现与审核，建议保留：

- `processed_data/pageindex/*.json`：PageIndex 树索引。
- `processed_data/pages/*.jsonl`：页级文本。
- `processed_data/catalog/doc_catalog.jsonl`：B 榜候选召回数据。
- `outputs/evidence.json`：每题证据定位。
- `outputs/logs/*.jsonl`：每次模型调用的 prompt 摘要、模型名、usage、qid、阶段。
- `requirements.txt`：包含 PageIndex requirements 与项目依赖。
- `README.md`：一键构建索引、一键跑题、一键导出结果。

日志中不要保存 API Key，不要保存与标准答案相关的外部信息。

## 24. 常见问题

### 24.1 PageIndex 生成的页码不准怎么办？

优先检查 PDF 抽取文本是否乱码、目录页是否被识别、标题是否被拆分。对高价值文档可改用 Markdown 路线。正式答题时可读取节点页段的前后页，并用证据抽取器判断具体证据是否存在。

### 24.2 没有摘要会不会影响树检索？

会有一定影响，但能降低成本和合规风险。可用标题、条款号、领域关键词、选项关键词做预筛。若摘要收益明显，使用 Qwen 离线生成摘要，并记录成本。

### 24.3 B 榜不用向量检索是否够用？

数据规模较小，按领域过滤后文档数量有限。元数据、标题、章节标题、关键词和 Qwen 候选选择通常足够构建强基线。重点是候选召回后必须用 PageIndex 页级证据验证。

### 24.4 是否应该让 Agent 自由调用工具？

不建议完全自由。选择题评测更适合确定性流水线：文档召回、节点召回、页读取、证据抽取、逐项判断。这样更容易控制 Token、调试错误和生成 evidence。

## 25. 最终检查清单

- [ ] PageIndex 作为 submodule，未修改源码。
- [ ] 正式推理阶段所有 LLM 调用均为 Qwen 系列。
- [ ] 非 Qwen 预处理结果不包含用于正式答题的语义摘要/FAQ/召回结果。
- [ ] A 榜直接使用题目 `doc_ids`，B 榜有候选文档召回流程。
- [ ] 每题 evidence 可追溯到 `doc_id + node_id + pages + quote`。
- [ ] 多选答案已去重、排序、无分隔符。
- [ ] Token usage 覆盖正式检索、压缩、判断、生成、自检。
- [ ] PageIndex 索引质量有离线检查和回退方案。
- [ ] `answer.csv` 包含 `summary` 行。
- [ ] 代码可一键复现生成 `answer.csv` 和 `evidence.json`。

## 26. 推荐落地方案

优先采用“Markdown PageIndex + Qwen 树检索 + 页级证据记忆”的组合：

1. 用高质量解析工具把 PDF 转为 Markdown 和页级文本。
2. 用 `md_to_tree()` 构建无摘要树，保留 `title/node_id/line_num`；PDF 页码另由页缓存维护。
3. 对 A 榜直接使用 `doc_ids`；对 B 榜用 doc catalog + Qwen 选择候选文档。
4. 对每个候选文档运行 PageIndex 树检索，选择少量节点。
5. 读取紧凑页段，按选项抽取证据。
6. 用证据记忆和程序化计算完成最终判断。
7. 输出标准答案和可审核证据。

该方案最符合赛题目标：减少全文输入、提高证据定位能力、保留可追溯性，并把 PageIndex 控制在第三方库职责范围内。
