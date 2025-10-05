// Source-based slice around line 88
// Method: <com.google.common.graph.Graphs: boolean hasCycle(Network)>

  }

  /**
   * Returns true if {@code network} has at least one cycle. A cycle is defined as a non-empty
   * subset of edges in a graph arranged to form a path (a sequence of adjacent outgoing edges)
   * starting and ending with the same node.
   *
   * <p>This method will detect any non-empty cycle, including self-loops (a cycle of length 1).
   */
  public static boolean hasCycle(Network<?, ?> network) {
    // In a directed graph, parallel edges cannot introduce a cycle in an acyclic graph.
    // However, in an undirected graph, any parallel edge induces a cycle in the graph.
    if (!network.isDirected()
        && network.allowsParallelEdges()
        && network.edges().size() > network.asGraph().edges().size()) {
      return true;
    }
    return hasCycle(network.asGraph());
  }

