// Source-based slice around line 615
// Method: <com.google.common.graph.Graphs: MutableValueGraph copyOf(ValueGraph)>

      copy.addNode(node);
    }
    for (EndpointPair<N> edge : graph.edges()) {
      copy.putEdge(edge.nodeU(), edge.nodeV());
    }
    return copy;
  }

  /** Creates a mutable copy of {@code graph} with the same nodes, edges, and edge values. */
  public static <N, V> MutableValueGraph<N, V> copyOf(ValueGraph<N, V> graph) {
    MutableValueGraph<N, V> copy =
        ValueGraphBuilder.from(graph).expectedNodeCount(graph.nodes().size()).build();
    for (N node : graph.nodes()) {
      copy.addNode(node);
    }
    for (EndpointPair<N> edge : graph.edges()) {
      // requireNonNull is safe because the endpoint pair comes from the graph.
      copy.putEdgeValue(
          edge.nodeU(),
          edge.nodeV(),
