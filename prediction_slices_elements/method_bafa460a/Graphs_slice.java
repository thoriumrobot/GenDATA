// Source-based slice around line 62
// Method: <com.google.common.graph.Graphs: boolean hasCycle(Graph)>

  // Graph query methods

  /**
   * Returns true if {@code graph} has at least one cycle. A cycle is defined as a non-empty subset
   * of edges in a graph arranged to form a path (a sequence of adjacent outgoing edges) starting
   * and ending with the same node.
   *
   * <p>This method will detect any non-empty cycle, including self-loops (a cycle of length 1).
   */
  public static <N> boolean hasCycle(Graph<N> graph) {
    int numEdges = graph.edges().size();
    if (numEdges == 0) {
      return false; // An edge-free graph is acyclic by definition.
    }
    if (!graph.isDirected() && numEdges >= graph.nodes().size()) {
      return true; // Optimization for the undirected case: at least one cycle must exist.
    }

    Map<Object, NodeVisitState> visitedNodes =
        Maps.newHashMapWithExpectedSize(graph.nodes().size());
