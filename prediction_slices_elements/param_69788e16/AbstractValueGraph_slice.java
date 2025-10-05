// Source-based slice around line 154
// Method: <com.google.common.graph.AbstractValueGraph: Map edgeValueMap(ValueGraph)>

        + isDirected()
        + ", allowsSelfLoops: "
        + allowsSelfLoops()
        + ", nodes: "
        + nodes()
        + ", edges: "
        + edgeValueMap(this);
  }

  private static <N, V> Map<EndpointPair<N>, V> edgeValueMap(ValueGraph<N, V> graph) {
    return Maps.asMap(
        graph.edges(),
        edge ->
            // requireNonNull is safe because the endpoint pair comes from the graph.
            requireNonNull(graph.edgeValueOrDefault(edge.nodeU(), edge.nodeV(), null)));
  }
}
