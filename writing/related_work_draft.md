# Related Work

## Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) was introduced by Lewis et al. (2020) as a framework that
conditions a seq2seq language model on documents retrieved from a non-parametric memory at
inference time. By interleaving retrieval with generation, RAG systems achieve strong performance
on knowledge-intensive NLP tasks without requiring the entire world's knowledge to be stored in
model parameters. The retrieval component in the original formulation relies on Dense Passage
Retrieval (DPR; Karpukhin et al., 2020), which encodes queries and passages with separate BERT
encoders and retrieves via maximum inner-product search over a FAISS index. Subsequent work has
refined this pipeline along multiple axes: bi-encoder fine-tuning (Izacard and Grave, 2021),
iterative retrieval (Shao et al., 2023), and query reformulation (Mao et al., 2021). HyperRAG
inherits the DPR-style dense retrieval component but replaces flat top-K selection with a
graph-guided expansion step.

## HTML-Aware Retrieval

Web-sourced corpora present challenges for standard RAG pipelines because documents are natively
encoded in HTML, which carries semantic structure — headings, tables, lists — that plain-text
extraction discards. HtmlRAG (Tan et al., 2024) addresses this by retaining cleaned HTML rather
than stripped plaintext as the retrieval unit, demonstrating consistent gains on open-domain QA
benchmarks. The AXE framework (2026) extends this direction by learning which HTML subtrees are
most relevant for a given query, effectively performing within-document retrieval at the tag level.
Our B2 baseline follows the HtmlRAG approach, preserving `<h1>`, `<h2>`, `<p>`, `<table>`, and
`<li>` tags while stripping non-informative elements. HyperRAG further builds on this intuition:
the hyperlink anchors embedded in HTML are not merely navigation artefacts but encode topical
relatedness that can be exploited as a zero-cost graph signal.

## Graph-Augmented Retrieval

Several recent systems augment retrieval with explicit graph structures. GraphRAG (Edge et al.,
2024) constructs a community-detection graph over extracted entity mentions, then summarises graph
communities as context for question answering. This approach yields strong results on queries
requiring synthesis across many documents but incurs significant pre-processing cost: entity
extraction, relation linking, and community detection must be performed over the entire corpus
before any query is issued. RAPTOR (Sarthi et al., 2024) takes a hierarchical clustering approach,
recursively summarising document chunks into tree nodes that encode multi-granularity context.
Both GraphRAG and RAPTOR build graphs from *derived* representations — entity co-occurrences or
embedding-space proximity — rather than from the structural signals already present in the source
documents. HyperRAG, by contrast, uses the hyperlink graph natively embedded in Wikipedia HTML,
requiring no entity extraction or clustering: graph edges are read directly from `<a href>` anchors
during corpus construction.

## Multi-Hop Question Answering

Multi-hop QA requires aggregating evidence across two or more documents to answer a question.
HotpotQA (Yang et al., 2018) is the standard benchmark for this task: its `fullwiki` setting
provides questions annotated with supporting facts drawn from English Wikipedia, and answer
strings that require reasoning over at least two pages. Subsequent benchmarks have raised the
difficulty further (MuSiQue, 2MWiki), but HotpotQA remains the most widely reported reference
point. On the systems side, IRCoT (Trivedi et al., 2023) interleaves retrieval and chain-of-thought
reasoning steps, each retrieval query conditioned on the reasoning trace produced so far. This
iterative approach is powerful but presupposes a capable language model at inference time.
For Milestone 2, HyperRAG targets the retrieval sub-problem only: given a multi-hop question,
can a single round of graph-expanded retrieval surface the relevant supporting pages without
iterative reasoning?

## The Gap

Despite the breadth of work across RAG, HTML-aware retrieval, and graph-augmented systems, no
prior method exploits the *native hyperlink structure* of web documents as a graph signal for
multi-hop retrieval. GraphRAG constructs graphs from extracted entities; HtmlRAG uses HTML
structure within a single document; DPR and its successors operate over flat passage sets. The
hyperlink graph is effectively free — it is already present in any HTML corpus — and it directly
encodes the editorial judgement of Wikipedia authors about which pages are topically related.
HyperRAG bridges this gap: by combining dense retrieval with one-hop hyperlink expansion and
cosine re-ranking, it recovers multi-hop supporting evidence that flat retrieval misses, without
any additional annotation, entity extraction, or iterative model calls.

---

## References

- Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for
  knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense passage retrieval for open-domain
  question answering. *EMNLP 2020*.
- Tan, Y., et al. (2024). HtmlRAG: HTML is better than plain text for modeling retrieved
  web knowledge in RAG systems. *arXiv:2411.02959*.
- Edge, D., Trinh, H., Cheng, N., et al. (2024). From local to global: A graph RAG approach
  to query-focused summarization. *arXiv:2404.16130*.
- Yang, Z., Qi, P., Zhang, S., et al. (2018). HotpotQA: A dataset for diverse, explainable
  multi-hop question answering. *EMNLP 2018*.
- Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving retrieval
  with chain-of-thought reasoning for knowledge-intensive multi-step questions. *ACL 2023*.
- Sarthi, P., Abdullah, R., Tuli, A., et al. (2024). RAPTOR: Recursive abstractive processing
  for tree-organized retrieval. *ICLR 2024*.
