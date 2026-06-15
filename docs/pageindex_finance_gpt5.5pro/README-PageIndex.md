<div align="center">
  
<a href="https://vectify.ai/pageindex" target="_blank">
  <img src="https://github.com/user-attachments/assets/46201e72-675b-43bc-bfbd-081cc6b65a1d" alt="PageIndex Banner" />
</a>

<br/>
<br/>

<p align="center">
  <a href="https://trendshift.io/repositories/14736" target="_blank"><img src="https://trendshift.io/api/badge/repositories/14736" alt="VectifyAI%2FPageIndex | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# PageIndex：无向量、基于推理的 RAG

<p align="center"><b>基于推理的 RAG（Retrieval-Augmented Generation，检索增强生成）&nbsp; ◦ &nbsp;无需向量数据库，无需分块&nbsp; ◦ &nbsp;上下文感知检索&nbsp; ◦ &nbsp;类人化</b></p>

<h4 align="center">
  <a href="https://vectify.ai">🌐 官网</a>&nbsp; • &nbsp;
  <a href="https://chat.pageindex.ai">🖥️ 聊天平台</a>&nbsp; • &nbsp;
  <a href="https://pageindex.ai/developer">🔌 MCP 与 API</a>&nbsp; • &nbsp;
  <a href="https://docs.pageindex.ai">📖 文档</a>&nbsp; • &nbsp;
  <a href="https://discord.com/invite/VuXuf29EUj">💬 Discord</a>&nbsp; • &nbsp;
  <a href="https://ii2abc2jejf.typeform.com/to/tK3AXl8T">✉️ 联系我们</a>&nbsp;
</h4>
  
</div>


<details open>
<summary><h2>📢 更新</h2></summary>

- 🔥 [**Agentic Vectorless RAG（智能体式无向量 RAG）**](https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py) — 一个简单的智能体式、无向量 RAG [示例](#agentic-vectorless-rag-an-example)，使用*自托管 PageIndex* 和 OpenAI Agents SDK。
- [**将 PageIndex 扩展到数百万文档**](https://pageindex.ai/blog/pageindex-filesystem) — *PageIndex File System（PageIndex 文件系统）* 是一个文件级树层，使 PageIndex 能够对整个语料库进行推理，而不只是单个文档，从而支持大规模文档搜索。
- [PageIndex Chat](https://chat.pageindex.ai) — 面向专业长文档的类人化文档分析智能体[平台](https://chat.pageindex.ai)。也可通过 [MCP](https://pageindex.ai/developer) 或 [API](https://pageindex.ai/developer) 使用。
- [PageIndex Framework](https://pageindex.ai/blog/pageindex-intro) — 深入解析 PageIndex：一种*智能体式、上下文内树索引*，使大语言模型（LLM，Large Language Model）能够在长文档上执行*基于推理、上下文感知的检索*。

 <!-- **🧪 Cookbooks:**
- [Vectorless RAG](https://docs.pageindex.ai/cookbook/vectorless-rag-pageindex): A minimal, hands-on example of reasoning-based RAG using PageIndex. No vectors, no chunking, and human-like retrieval.
- [Vision-based Vectorless RAG](https://docs.pageindex.ai/cookbook/vision-rag-pageindex): OCR-free, vision-only RAG with PageIndex's reasoning-native retrieval workflow that works directly over PDF page images. -->

</details>

---

# 📑 PageIndex 简介

你是否对向量数据库在长篇专业文档上的检索准确率感到失望？传统的基于向量的 RAG 依赖语义*相似性*，而不是真正的*相关性*。但**相似性 ≠ 相关性**——检索中我们真正需要的是**相关性**，而这需要**推理**。在处理需要领域专业知识和多步推理的专业文档时，相似性搜索往往力不从心：它会遗漏相关但不相似的内容，也会返回相似但不相关的内容。

受 AlphaGo 启发，我们提出 **[PageIndex](https://vectify.ai/pageindex)** —— 一个**无向量**、**基于推理的 RAG** 系统。它从长文档构建**层级树索引**，并使用 LLM 在*该索引上进行推理*，实现**智能体式、上下文感知检索**。检索过程*可追踪*且*可解释*，无需向量数据库，也无需分块。
PageIndex 通过*树搜索*模拟*人类专家*浏览复杂文档并提取知识的方式，使 LLM 能够通过*思考*和*推理*找到最相关的文档章节。它分两步执行检索：

1. 生成文档的“目录式”**树结构索引**
2. 通过**树搜索**执行基于推理的检索

<div align="center">
  <a href="https://pageindex.ai/blog/pageindex-intro" target="_blank" title="The PageIndex Framework">
    <img src="https://docs.pageindex.ai/images/cookbook/vectorless-rag.png" width="70%">
  </a>
</div>

### 🎯 核心特性

与传统的基于向量的 RAG 相比，**PageIndex** 具有以下特性：
- **无需向量数据库**：使用文档结构和 LLM 推理进行检索，而不是向量相似性搜索。
- **无需分块**：文档被组织为自然章节，而不是人为切分的块。
- **更好的可追踪性与可解释性**：检索由推理驱动，并基于明确的页码和章节引用，使每个结果都可追踪、可解释——不再依赖不透明、近似的向量搜索进行“凭感觉检索”。
- **上下文感知检索**：检索依赖完整上下文（例如对话历史和领域知识），并且可以轻松纳入新的上下文。
- **类人化检索**：模拟人类专家浏览复杂文档并提取知识的方式。

PageIndex 支撑了一个基于推理的 RAG 系统，该系统在 FinanceBench 上达到**当前最佳** [98.7% 准确率](https://github.com/VectifyAI/Mafin2.5-FinanceBench)，在专业文档分析任务中大幅超越基于向量的 RAG 方案（见[博客文章](https://vectify.ai/blog/Mafin2.5)）。

### 📍 探索 PageIndex

如需了解更多，请阅读 [PageIndex 框架](https://pageindex.ai/blog/pageindex-intro)的详细介绍。你也可以查看[我们的 GitHub](https://docs.pageindex.ai/open-source) 获取开源代码，并参考 [cookbooks](https://docs.pageindex.ai/cookbook)、[教程](https://docs.pageindex.ai/tutorials)和[博客](https://pageindex.ai/blog)了解更多使用指南和示例。

PageIndex 服务可作为 ChatGPT 风格的[聊天平台](https://chat.pageindex.ai)使用，也可以通过 [MCP](https://pageindex.ai/developer) 或 [API](https://pageindex.ai/developer) 集成，并支持[企业级](https://pageindex.ai/enterprise)部署。

### 🛠️ 部署选项
- **自托管** — 使用本开源仓库在本地运行（采用标准 PDF 解析）。
- **云服务** — 生产级流水线，提供增强 OCR、树构建和检索能力，以获得最佳效果。可在我们的[聊天平台](https://chat.pageindex.ai/)即时试用，或通过 [MCP](https://pageindex.ai/developer) 或 [API](https://pageindex.ai/developer) 集成。
- **企业版** — 专用或私有部署（VPC、本地部署）。[联系我们](https://ii2abc2jejf.typeform.com/to/gVv7qkaN)或[预约演示](https://calendly.com/pageindex/meet)了解更多。

### 🧪 快速上手

- 🔥 [**Agentic Vectorless RAG（智能体式无向量 RAG）**](examples/agentic_vectorless_rag_demo.py)（**最新**）— 一个简单但完整的**智能体式无向量 RAG** [示例](#agentic-vectorless-rag-an-example)，使用*自托管* PageIndex 和 OpenAI Agents SDK。
- 试用 [Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb) 笔记本 —— 一个使用 PageIndex 构建基于推理的 RAG 的*最小化*动手示例。
- 查看 [Vision-based Vectorless RAG](https://github.com/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb) —— 无需 OCR；一个最小化的、基于视觉且原生支持推理的 RAG 流水线，可直接处理页面图像。
  
<div align="center">
  <a href="https://github.com/VectifyAI/PageIndex/blob/main/examples/agentic_vectorless_rag_demo.py" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/View_on_GitHub-Agentic_Vectorless_RAG-blue?style=for-the-badge&logo=github" alt="View on GitHub: Agentic Vectorless RAG" />
  </a>
  <br/>
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/pageindex_RAG_simple.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vectorless_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vectorless RAG" />
  </a>
  &nbsp;&nbsp;
  <a href="https://colab.research.google.com/github/VectifyAI/PageIndex/blob/main/cookbook/vision_RAG_pageindex.ipynb" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Open_In_Colab-Vision_RAG-orange?style=for-the-badge&logo=googlecolab" alt="Open in Colab: Vision RAG" />
  </a>
</div>

---

# 🌲 PageIndex 树结构

PageIndex 可以将长篇 PDF 文档转换为语义化的**树结构**，类似于_“目录”_，但针对大语言模型（LLM）的使用进行了优化。它非常适合：财务报告、监管申报文件、学术教材、法律或技术手册，以及任何超出 LLM 上下文长度限制的文档。

下面是一个 PageIndex 树结构示例。另请参阅更多示例[文档](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents)以及生成的[树结构](https://github.com/VectifyAI/PageIndex/tree/main/examples/documents/results)。

```jsonc
...
{
  "title": "Financial Stability",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 22,
  "summary": "The Federal Reserve ...",
  "nodes": [
    {
      "title": "Monitoring Financial Vulnerabilities",
      "node_id": "0007",
      "start_index": 22,
      "end_index": 28,
      "summary": "The Federal Reserve's monitoring ..."
    },
    {
      "title": "Domestic and International Cooperation and Coordination",
      "node_id": "0008",
      "start_index": 28,
      "end_index": 31,
      "summary": "In 2023, the Federal Reserve collaborated ..."
    }
  ]
}
...
```

你可以使用本开源仓库生成 PageIndex 树结构；也可以使用我们的 [API](https://pageindex.ai/developer)，通过增强 OCR 和树构建流水线获得更高质量的结果。

---

# ⚙️ 包使用方法

> **注意：** 本包使用标准 PDF 解析。对于复杂 PDF 用例，我们的[云服务](https://pageindex.ai/developer)（通过 MCP 和 API）提供增强 OCR、树构建和检索能力。

你可以按以下步骤从 PDF 文档生成 PageIndex 树。

### 1. 安装依赖

```bash
pip3 install --upgrade -r requirements.txt
```

### 2. 设置你的 LLM API 密钥

在根目录中创建 `.env` 文件，并写入你的 LLM API 密钥。通过 [LiteLLM](https://docs.litellm.ai/docs/providers) 支持多 LLM：

```bash
OPENAI_API_KEY=your_openai_key_here
```

### 3. 为你的 PDF 生成 PageIndex 结构

```bash
python3 run_pageindex.py --pdf_path /path/to/your/document.pdf
```

<details>
<summary>可选参数</summary>
<br>
你可以使用其他可选参数自定义处理过程：

```
--model                 要使用的 LLM 模型（默认：gpt-4o-2024-11-20）
--toc-check-pages       检查目录的页数（默认：20）
--max-pages-per-node    每个节点的最大页数（默认：10）
--max-tokens-per-node   每个节点的最大 token 数（默认：20000）
--if-add-node-id        添加节点 ID（yes/no，默认：yes）
--if-add-node-summary   添加节点摘要（yes/no，默认：yes）
--if-add-doc-description 添加文档描述（yes/no，默认：yes）
```
</details>

<details>
<summary>Markdown 支持</summary>
<br>
我们也为 PageIndex 提供 Markdown 支持。你可以使用 `--md_path` 标志为 Markdown 文件生成树结构。

```bash
python3 run_pageindex.py --md_path /path/to/your/document.md
```

> 注意：在该模式下，我们使用 "#" 来判断节点标题及其层级。例如，"##" 表示二级标题，"###" 表示三级标题，依此类推。请确保你的 Markdown 文件格式正确。如果你的 Markdown 文件是从 PDF 或 HTML 转换而来，我们不建议使用该模式，因为现有多数转换工具无法保留原始层级结构。相反，请使用我们的 [PageIndex OCR](https://pageindex.ai/blog/ocr)，它专为保留层级结构而设计，可先将 PDF 转换为 Markdown 文件，再使用该模式。
</details>

## Agentic Vectorless RAG：示例

如需一个使用**自托管 PageIndex**（结合 OpenAI Agents SDK）的简单端到端**智能体式无向量 RAG** 示例，请参阅 [`examples/agentic_vectorless_rag_demo.py`](examples/agentic_vectorless_rag_demo.py)。

```bash
# 安装可选依赖
pip3 install openai-agents

# 运行演示
python3 examples/agentic_vectorless_rag_demo.py
```

<!--
# ☁️ 使用 PageIndex OCR 改进树生成

本仓库旨在为简单 PDF 生成 PageIndex 树结构，但许多真实场景涉及复杂 PDF，传统 Python 工具难以解析。然而，从 PDF 文档中提取高质量文本仍然是一个并不简单的挑战。多数 OCR 工具只能提取页面级内容，会丢失更广泛的文档上下文和层级结构。

为解决这一问题，我们推出了 PageIndex OCR —— 首个旨在保留文档全局结构的长上下文 OCR 模型。PageIndex OCR 在识别跨文档页面的真实层级和语义关系方面，显著优于来自 Mistral 和 Contextual AI 等机构的领先 OCR 工具。

- 在我们的 [Dashboard](https://dash.pageindex.ai/) 体验更高水平的 OCR 质量。
- 通过我们的 [API](https://docs.pageindex.ai/quickstart) 将 PageIndex OCR 无缝集成到你的技术栈中。

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb35d8ae-865c-4e60-a33b-ebbd00c41732" width="80%">
</p>
-->

---

# 📈 案例研究：PageIndex 领先 Finance QA 基准

[Mafin 2.5](https://vectify.ai/mafin) 是一个用于金融文档分析的基于推理的 RAG 系统，由 **PageIndex** 提供支持。它在 [FinanceBench](https://arxiv.org/abs/2311.11944) 基准上达到当前最佳的 [**98.7% 准确率**](https://vectify.ai/blog/Mafin2.5)，显著超越传统的基于向量的 RAG 系统。

PageIndex 的层级索引和推理驱动检索，能够从复杂金融报告（如 SEC 申报文件和业绩披露）中精准定位并提取相关上下文。

查看完整[基准结果](https://github.com/VectifyAI/Mafin2.5-FinanceBench)和我们的[博客文章](https://vectify.ai/blog/Mafin2.5)，了解详细比较和性能指标。

<div align="center">
  <a href="https://github.com/VectifyAI/Mafin2.5-FinanceBench">
    <img src="https://github.com/user-attachments/assets/571aa074-d803-43c7-80c4-a04254b782a3" width="70%">
  </a>
</div>

---

# 🧭 资源

* 📝 [博客](https://pageindex.ai/blog)：技术文章、研究洞察和产品更新。
* 🔧 [开发者](https://pageindex.ai/developer)：MCP 设置、API 文档和集成指南。
* 🧪 [Cookbooks](https://docs.pageindex.ai/cookbook)：动手可运行示例和高级用例。
* 📖 [教程](https://docs.pageindex.ai/tutorials)：实用指南和策略，包括*文档搜索*和*树搜索*。

---

# ⭐ 支持我们

如果你喜欢我们的项目，请给我们一个 star 🌟。谢谢！  

<p>
  <img src="https://github.com/user-attachments/assets/eae4ff38-48ae-4a7c-b19f-eab81201d794" width="80%">
</p>

引用本工作请使用：
```
Mingtian Zhang, Yu Tang and PageIndex Team,
"PageIndex: Next-Generation Vectorless, Reasoning-based RAG",
PageIndex Blog, Sep 2025.
```

<details>
<summary>或使用 BibTeX 引用。</summary>

```bibtex
@article{zhang2025pageindex,
  author = {Mingtian Zhang and Yu Tang and PageIndex Team},
  title = {PageIndex: Next-Generation Vectorless, Reasoning-based RAG},
  journal = {PageIndex Blog},
  year = {2025},
  month = {September},
  note = {https://pageindex.ai/blog/pageindex-intro},
}
```
</details>


### 🌐 生态系统

PageIndex 生态中的其他[开源项目](https://docs.pageindex.ai/open-source)：[OpenKB](https://github.com/VectifyAI/OpenKB) 是一个 LLM 知识库，可将文档编译为相互链接的 wiki。[ChatIndex](https://github.com/VectifyAI/ChatIndex) 将树索引和检索扩展到长对话历史。[ConDB](https://github.com/VectifyAI/ConDB) 是一个 KV-cache 原生的上下文数据库，用于基于树的检索。[PageIndex MCP](https://github.com/VectifyAI/pageindex-mcp) 是 PageIndex 的 MCP 服务器。

### 联系我们

<div align="center">

[![Website](https://img.shields.io/badge/Website-2D72CF?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDEgMSAxMWgyLjV2MTJoNnYtN2g1djdoNlYxMUgyM3oiLz48L3N2Zz4%3D)](https://pageindex.ai)&nbsp;
[![Twitter](https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/PageIndexAI)&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgNCAyMCAxNiI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwLjQ1IDIwLjQ1aC0zLjU1di01LjU3YzAtMS4zMy0uMDMtMy4wNC0xLjg1LTMuMDQtMS44NSAwLTIuMTQgMS40NS0yLjE0IDIuOTR2NS42N0g5LjM1VjloMy40MXYxLjU2aC4wNWMuNDgtLjkgMS42NC0xLjg1IDMuMzctMS44NSAzLjYgMCA0LjI3IDIuMzcgNC4yNyA1LjQ2djYuMjh6TTUuMzQgNy40M2EyLjA2IDIuMDYgMCAxIDEgMC00LjEzIDIuMDYgMi4wNiAwIDAgMSAwIDQuMTN6TTcuMTIgMjAuNDVIMy41NlY5aDMuNTZ2MTEuNDV6TTIyLjIyIDBIMS43N0MuNzkgMCAwIC43NyAwIDEuNzN2MjAuNTRDMCAyMy4yMy43OSAyNCAxLjc3IDI0aDIwLjQ1QzIzLjIgMjQgMjQgMjMuMjMgMjQgMjIuMjdWMS43M0MyNCAuNzcgMjMuMiAwIDIyLjIyIDB6Ii8%2BPC9zdmc%2B)](https://www.linkedin.com/company/vectify-ai/)&nbsp;
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.com/invite/VuXuf29EUj)&nbsp;
[![Book a Demo](https://img.shields.io/badge/Book_a_Demo-6E7E96?style=for-the-badge&logo=googlecalendar&logoColor=white)](https://calendly.com/pageindex/meet)&nbsp;
[![Contact Us](https://img.shields.io/badge/Contact_Us-3B82F6?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgNCAyMCAxNiI%2BPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTIwIDRINGMtMS4xIDAtMiAuOS0yIDJ2MTJjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0wIDQtOCA1LTgtNVY2bDggNSA4LTV6Ii8%2BPC9zdmc%2B)](https://ii2abc2jejf.typeform.com/to/tK3AXl8T)

</div>

---

© 2026 [Vectify AI](https://vectify.ai)
