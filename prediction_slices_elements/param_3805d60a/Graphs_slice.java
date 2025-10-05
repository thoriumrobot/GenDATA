// Source-based slice around line 603
// Method: <com.google.common.graph.Graphs: MutableGraph copyOf(Graph)>

        if (subgraph.nodes().contains(successorNode)) {
          subgraph.addEdge(node, successorNode, edge);
        }
      }
    }
    return subgraph;
  }

  /** Creates a mutable copy of {@code graph} with the same nodes and edges. */
  public static <N> MutableGraph<N> copyOf(Graph<N> graph) {
    MutableGraph<N> copy = GraphBuilder.from(graph).expectedNodeCount(graph.nodes().size()).build();
    for (N node : graph.nodes()) {
      copy.addNode(node);
    }
    for (EndpointPair<N> edge : graph.edges()) {
      copy.putEdge(edge.nodeU(), edge.nodeV());
    }
    return copy;
  }

