// Source-based slice around line 84
// Method: <com.google.common.graph.ImmutableValueGraph: ImmutableMap getNodeConnections(ValueGraph)>

  public ElementOrder<N> incidentEdgeOrder() {
    return ElementOrder.stable();
  }

  @Override
  public ImmutableGraph<N> asGraph() {
    return new ImmutableGraph<>(this); // safe because the view is effectively immutable
  }

  private static <N, V> ImmutableMap<N, GraphConnections<N, V>> getNodeConnections(
      ValueGraph<N, V> graph) {
    // ImmutableMap.Builder maintains the order of the elements as inserted, so the map will have
    // whatever ordering the graph's nodes do, so ImmutableSortedMap is unnecessary even if the
    // input nodes are sorted.
    ImmutableMap.Builder<N, GraphConnections<N, V>> nodeConnections = ImmutableMap.builder();
    for (N node : graph.nodes()) {
      nodeConnections.put(node, connectionsOf(graph, node));
    }
    return nodeConnections.buildOrThrow();
  }
