结论：**PageIndex 可以运用到这个赛题，但更适合作为“结构化导航索引 + 证据定位框架”，不建议直接使用其官方 Chat/API/MCP 作为答题黑盒。**最稳妥方案是：自托管 PageIndex 思路，把所有 LLM 调用替换为 Qwen 系列模型，并把 PageIndex 输出限制在“章节树、页码范围、标题、结构化定位”层面，再由 Qwen 完成正式检索、证据判断和答案生成。

PageIndex 的核心是“无向量、推理式 RAG（Vectorless, Reasoning-based RAG）”：先为长文档生成类似目录的层级树，再让 LLM 沿树搜索相关章节，而不是用向量相似度直接召回 chunk。官方 README 明确说它不依赖向量库、不做传统 chunking，并通过树搜索做可追溯的上下文检索；这与赛题的长金融文档、多章节引用、证据定位、Token 控制高度匹配。([GitHub][1])

它尤其适合赛题里的三类问题：监管法规、保险条款、金融合同。这些文档通常有条、款、章、节、附录、定义、例外条件，PageIndex 的“目录树/章节树（ToC Tree）”可以天然承载这些层级。官方说明中，PageIndex 会把 PDF 转成语义树结构，节点包含标题、起止页、摘要等字段，并能映射回原始内容。([GitHub][1]) 赛题也明确要求系统处理跨文档定位、复杂推理、动态记忆压缩和 Token 成本优化，且文档包括保险、监管、金融合同、财报、研报五类金融长文本。

但直接套用会有合规风险。赛题要求推理问答阶段的文档定位、段落检索、记忆压缩、证据判断、答案生成都必须严格使用 Qwen 系列模型；并且非 Qwen 预处理能力不能延伸到正式检索、rerank、召回、纠错或答案投票。 PageIndex 默认示例使用 OpenAI API key，README 中默认模型是 `gpt-4o-2024-11-20`，并且其 agentic 示例使用 OpenAI Agents SDK；这些默认链路不能直接用于正式答题。([GitHub][1])

可行的改造方式是：使用自托管 PageIndex 代码，而不是 PageIndex 云服务。PageIndex README 提到 self-host 可以本地运行，云服务则带增强 OCR、tree building、retrieval；在赛题中，云端 PageIndex 的隐藏模型和检索过程不可控，不利于代码审核和模型限制合规。([GitHub][1]) 好消息是 PageIndex 依赖 LiteLLM，README 也写了 Multi-LLM supported via LiteLLM；LiteLLM 官方文档支持 DashScope/Qwen 模型，阿里云 Model Studio 也支持 OpenAI-compatible 接口，所以理论上可以把 PageIndex 的 LLM backend 改成 `dashscope/qwen-*` 或百炼兼容接口。([GitHub][1])

建议采用这样的落地架构：

第一层是文档解析。用 MinerU、PyMuPDF、pdfplumber、Docling 等把 PDF 解析成 Markdown/JSON，保留页码、标题、条款编号、表格、脚注、附录。PageIndex 自托管包使用的是标准 PDF parsing，官方也提示复杂 PDF 更适合增强 OCR；金融年报和研报表格很多，所以不要完全依赖 PageIndex 原生 PDF 解析。([GitHub][1])

第二层是 PageIndex 风格的结构树。为每个 doc_id 生成一棵 `doc_tree.json`，节点保留 `doc_id / node_id / title / start_page / end_page / heading_path / parent / children`。我建议**关闭或弱化 node summary（节点摘要）**，至少不要用非 Qwen 生成的语义摘要参与正式检索；如果确实需要摘要，应由 Qwen 生成，并计入 Token 统计。PageIndex 默认支持 node summary 和 doc description 选项，但这正是赛题边界最敏感的地方。([GitHub][1])

第三层是 A 榜检索。A 榜给 doc_ids，直接在指定文档树里做树搜索：先让 Qwen 解析题干和选项，抽取实体、指标、条款号、年份、金额、比例、关键词；再在 PageIndex 树上选择候选节点；然后只读取候选节点对应的原文页段，交给 Qwen 逐选项判断。这样能显著减少全文输入，符合赛题对 Token 效率的要求。赛题评分中 Token 效率最多影响 30% 加权，且统计覆盖检索摘要、上下文压缩、证据判断、答案生成和自检等所有模型调用。

第四层是 B 榜检索。PageIndex 官方教程说明，它默认更偏单文档 reasoning-based RAG，多文档搜索需要额外工作流，如按 metadata、semantics、description 搜索。([GitHub][2]) 对赛题 B 榜，不能只靠 PageIndex 单文档树；需要先做 corpus-level 候选文档选择。建议用“领域 + 标题/年份/公司名/法规名/产品名/条款号/BM25/关键词倒排/结构化 metadata”先召回 3–8 个候选文档，再对每个候选文档跑 PageIndex 树内定位。避免使用非 Qwen embedding（嵌入）或非 Qwen reranker（重排序器），否则会碰到模型限制。

第五层是证据聚合和答案约束。每个选项至少保留 1–3 条证据，输出 `evidence.json`，包括 `qid、option、doc_id、node_id、page_range、quote、verdict、reason`。最后答案只输出合法大写字母；多选题必须去重并按字母排序，因为赛题要求多选答案完全匹配，不给部分分。

适用性判断：

PageIndex 对“法规条文定位、保险责任触发条件、债券募集说明书条款、年报章节导航、研报观点核验”很有价值。它的强项是让模型知道“该去哪里找”，尤其适合章节结构强、跨引用多、原文证据必须可追溯的文档。官方也强调其可追溯、可解释、基于页码/章节引用的 retrieval。([GitHub][1])

PageIndex 不足以单独解决“数值计算、表格还原、跨文档指标对比、B 榜盲检索、答案校验”。这些还需要表格结构化、规则计算器、候选文档检索、Qwen 逐项判别和后处理。特别是财报问题经常要求营业收入、净利润、现金流、研发投入占比的跨年对比；PageIndex 能帮你定位相关页，但不能替代精确表格抽取和计算。

最终建议：**用 PageIndex，但只用其思想和自托管组件，不用官方黑盒服务；把它改造成 Qwen-only、可审计、证据可回溯的结构化检索模块。**在方案中可以写成“PageIndex-style hierarchical document navigation（PageIndex 风格层级文档导航）”，核心收益是降低长文档输入 Token、提高证据定位稳定性；核心风险是默认模型、node summary、云服务、多文档检索链路需要合规改造。

[1]: https://github.com/VectifyAI/PageIndex "GitHub - VectifyAI/PageIndex:  PageIndex: Document Index for Vectorless, Reasoning-based RAG · GitHub"
[2]: https://github.com/VectifyAI/PageIndex/blob/main/examples/tutorials/doc-search/README.md "PageIndex/examples/tutorials/doc-search/README.md at main · VectifyAI/PageIndex · GitHub"
