// Source-based slice around line 168
// Method: <com.google.common.graph.StandardValueGraph: boolean containsNode(N)>

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
  }

  private final boolean hasEdgeConnectingInternal(N nodeU, N nodeV) {
    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
    return (connectionsU != null) && connectionsU.successors().contains(nodeV);
  }

  private final @Nullable V edgeValueOrDefaultInternal(N nodeU, N nodeV, @Nullable V defaultValue) {
    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
