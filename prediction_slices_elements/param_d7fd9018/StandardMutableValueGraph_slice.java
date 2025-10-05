// Source-based slice around line 87
// Method: <com.google.common.graph.StandardMutableValueGraph: V putEdgeValue(N,N,V)>

  @CanIgnoreReturnValue
  private GraphConnections<N, V> addNodeInternal(N node) {
    GraphConnections<N, V> connections = newConnections();
    checkState(nodeConnections.put(node, connections) == null);
    return connections;
  }

  @Override
  @CanIgnoreReturnValue
  public @Nullable V putEdgeValue(N nodeU, N nodeV, V value) {
    checkNotNull(nodeU, "nodeU");
    checkNotNull(nodeV, "nodeV");
    checkNotNull(value, "value");

    if (!allowsSelfLoops()) {
      checkArgument(!nodeU.equals(nodeV), SELF_LOOPS_NOT_ALLOWED, nodeU);
    }

    GraphConnections<N, V> connectionsU = nodeConnections.get(nodeU);
    if (connectionsU == null) {
