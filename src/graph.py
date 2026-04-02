"""Graph module: NetworkX hyperlink graph construction and persistence."""

from pathlib import Path
from typing import Any

import networkx as nx


DEFAULT_GRAPH = Path("data/hyperlink_graph.graphml")


def build_graph(corpus: list[dict[str, Any]]) -> nx.DiGraph:
    """Build a directed graph from corpus hyperlinks.

    Nodes = Wikipedia page titles, edges = hyperlinks between corpus pages.

    Args:
        corpus: List of page dicts with 'title' and 'links' keys.

    Returns:
        NetworkX DiGraph.
    """
    raise NotImplementedError("Implemented in Phase 3")


def save_graph(G: nx.DiGraph, path: str | Path = DEFAULT_GRAPH) -> None:
    """Save graph to GraphML file.

    Args:
        G: NetworkX DiGraph to save.
        path: Output file path.
    """
    raise NotImplementedError("Implemented in Phase 3")


def load_graph(path: str | Path = DEFAULT_GRAPH) -> nx.DiGraph:
    """Load graph from GraphML file.

    Args:
        path: Path to GraphML file.

    Returns:
        NetworkX DiGraph.
    """
    raise NotImplementedError("Implemented in Phase 3")


def print_graph_stats(G: nx.DiGraph) -> None:
    """Print graph statistics: nodes, edges, avg degree, top-5 connected pages.

    Args:
        G: NetworkX DiGraph to analyze.
    """
    raise NotImplementedError("Implemented in Phase 3")
