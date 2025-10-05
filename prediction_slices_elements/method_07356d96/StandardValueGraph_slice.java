// Source-based slice around line 159
// Method: <com.google.common.graph.StandardValueGraph: GraphConnections checkedConnections(N)>

    validateEndpoints(endpoints);
    return edgeValueOrDefaultInternal(endpoints.nodeU(), endpoints.nodeV(), defaultValue);
  }

  @Override
  protected long edgeCount() {
    return edgeCount;
  }

  private final GraphConnections<N, V> checkedConnections(N node) {
    GraphConnections<N, V> connections = nodeConnections.get(node);
    if (connections == null) {
      checkNotNull(node);
      throw new IllegalArgumentException("Node " + node + " is not an element of this graph.");
    }
    return connections;
  }

  final boolean containsNode(@Nullable N node) {
    return nodeConnections.containsKey(node);
