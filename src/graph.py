"""Graph module: NetworkX hyperlink graph construction and persistence."""

import urllib.parse
from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_GRAPH = Path("data/hyperlink_graph.graphml")


def build_graph(corpus: list[dict[str, Any]]) -> nx.DiGraph:
    """Build a directed graph from corpus hyperlinks.

    Nodes = Wikipedia page titles, edges = hyperlinks between corpus pages.
    Uses two-pass construction: first builds a set of all corpus titles for O(1)
    membership checks, then adds edges only for within-corpus links.

    Links in page["links"] are already decoded human-readable titles (unquoted,
    underscores replaced with spaces) — corpus.py handles decoding at fetch time.
    Do NOT apply urllib.parse.unquote() to page["links"] here.

    Args:
        corpus: List of page dicts with keys: title, text, html, links.

    Returns:
        NetworkX DiGraph with nodes = page titles and edges = within-corpus links.
    """
    G: nx.DiGraph = nx.DiGraph()

    # Pass 1: Add all corpus pages as nodes with attributes
    corpus_titles: set[str] = set()
    for page in corpus:
        title: str = page["title"]
        url: str = (
            "https://en.wikipedia.org/wiki/"
            + urllib.parse.quote(title.replace(" ", "_"), safe="")
        )
        G.add_node(title, title=title, url=url)
        corpus_titles.add(title)

    # Pass 2: Add edges for within-corpus hyperlinks (skip self-loops and external links)
    for page in corpus:
        source: str = page["title"]
        for link in page.get("links", []):
            if link in corpus_titles and link != source:
                G.add_edge(source, link)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def save_graph(G: nx.DiGraph, path: str | Path = DEFAULT_GRAPH) -> None:
    """Save graph to GraphML file.

    Creates parent directories if they do not exist.

    Args:
        G: NetworkX DiGraph to save.
        path: Output file path (default: data/hyperlink_graph.graphml).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, path)
    print(f"Graph saved to {path}")


def load_graph(path: str | Path = DEFAULT_GRAPH) -> nx.DiGraph:
    """Load graph from GraphML file.

    Args:
        path: Path to GraphML file (default: data/hyperlink_graph.graphml).

    Returns:
        NetworkX DiGraph loaded from the file.
    """
    path = Path(path)
    return nx.read_graphml(path)


def print_graph_stats(G: nx.DiGraph) -> None:
    """Print graph statistics: nodes, edges, avg out-degree, top-5 by degree.

    Output format:
        Graph stats:
          Nodes: {n}
          Edges: {e}
          Avg out-degree: {avg:.2f}

        Top-5 most-cited pages (in-degree):
          1. {title} -- {in_degree} incoming links

        Top-5 most-linked pages (out-degree):
          1. {title} -- {out_degree} outgoing links

    Args:
        G: NetworkX DiGraph to analyze.
    """
    n: int = G.number_of_nodes()
    e: int = G.number_of_edges()
    avg_out: float = sum(d for _, d in G.out_degree()) / n if n > 0 else 0.0

    print("Graph stats:")
    print(f"  Nodes: {n}")
    print(f"  Edges: {e}")
    print(f"  Avg out-degree: {avg_out:.2f}")
    print()

    top_in = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:5]
    print("Top-5 most-cited pages (in-degree):")
    for rank, (title, in_degree) in enumerate(top_in, start=1):
        print(f"  {rank}. {title} -- {in_degree} incoming links")
    print()

    top_out = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:5]
    print("Top-5 most-linked pages (out-degree):")
    for rank, (title, out_degree) in enumerate(top_out, start=1):
        print(f"  {rank}. {title} -- {out_degree} outgoing links")


def build_and_save_graph(
    corpus: list[dict[str, Any]],
    output_path: str | Path = DEFAULT_GRAPH,
    force: bool = False,
) -> nx.DiGraph:
    """Build graph with cache-first pattern. Skips rebuild if output exists.

    Pipeline:
        1. If output_path exists and force=False, load and return cached graph.
        2. Otherwise, build graph from corpus using build_graph().
        3. Save to output_path using save_graph().
        4. Return the graph.

    Args:
        corpus: List of page dicts from corpus.py.
        output_path: Destination path for GraphML file (default: data/hyperlink_graph.graphml).
        force: If True, ignore cache and rebuild from scratch.

    Returns:
        NetworkX DiGraph (either loaded from cache or freshly built).
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"Loading cached graph from {output_path}")
        return load_graph(output_path)
    G = build_graph(corpus)
    save_graph(G, output_path)
    return G
