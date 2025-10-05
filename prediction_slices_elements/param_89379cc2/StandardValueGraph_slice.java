// Source-based slice around line 177
// Method: <com.google.common.graph.StandardValueGraph: V edgeValueOrDefaultInternal(N,N,V)>

  final boolean containsNode(@Nullable N node) {
    return nodeConnections.containsKey(node);
  }

  private final boolean hasEdgeConnectingInternal(N nodeU, N nodeV) {
    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
    return (connectionsU != null) && connectionsU.successors().contains(nodeV);
  }

  private final @Nullable V edgeValueOrDefaultInternal(N nodeU, N nodeV, @Nullable V defaultValue) {
    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
    V value = (connectionsU == null) ? null : connectionsU.value(nodeV);
    // TODO(b/192579700): Use a ternary once it no longer confuses our nullness checker.
    if (value == null) {
      return defaultValue;
    } else {
      return value;
    }
  }
}
