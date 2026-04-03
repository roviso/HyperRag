"""Tests for src/graph.py — NetworkX hyperlink graph construction."""

import networkx as nx
import pytest

from src.graph import build_graph, save_graph, load_graph, print_graph_stats


def test_build_graph_structure(sample_corpus_for_graph):
    """GRAPH-01: build_graph creates DiGraph with correct nodes, edges, and attributes."""
    G = build_graph(sample_corpus_for_graph)

    # Correct type
    assert isinstance(G, nx.DiGraph)

    # 3 corpus pages = 3 nodes
    assert G.number_of_nodes() == 3

    # Expected edges: Einstein->Bohr, Einstein->QM, Bohr->QM, Bohr->Einstein, QM->Einstein
    # "Nonexistent Page" is NOT in corpus so no edge for it
    assert G.number_of_edges() == 5

    # No self-loops
    assert not any(u == v for u, v in G.edges())

    # Node attributes present
    for node in G.nodes():
        assert "title" in G.nodes[node]
        assert "url" in G.nodes[node]
        assert G.nodes[node]["url"].startswith("https://en.wikipedia.org/wiki/")

    # Specific edge check
    assert G.has_edge("Albert Einstein", "Niels Bohr")
    assert not G.has_edge("Albert Einstein", "Nonexistent Page")


def test_save_load_roundtrip(sample_graph, tmp_path):
    """GRAPH-02: GraphML save/load preserves nodes, edges, and attributes."""
    filepath = tmp_path / "test_graph.graphml"
    save_graph(sample_graph, filepath)
    assert filepath.exists()

    loaded = load_graph(filepath)
    assert isinstance(loaded, nx.DiGraph)
    assert loaded.number_of_nodes() == sample_graph.number_of_nodes()
    assert loaded.number_of_edges() == sample_graph.number_of_edges()

    # Attributes preserved
    for node in sample_graph.nodes():
        assert node in loaded.nodes()
        assert loaded.nodes[node]["title"] == sample_graph.nodes[node]["title"]
        assert loaded.nodes[node]["url"] == sample_graph.nodes[node]["url"]


def test_graph_stats_runs(sample_graph, capsys):
    """GRAPH-03: print_graph_stats runs without error and prints expected format."""
    print_graph_stats(sample_graph)
    captured = capsys.readouterr()
    assert "Graph stats:" in captured.out
    assert "Nodes: 3" in captured.out
    assert "Edges: 2" in captured.out
    assert "Avg out-degree:" in captured.out
    assert "Top-5 most-cited pages (in-degree):" in captured.out
    assert "Top-5 most-linked pages (out-degree):" in captured.out
