// Source-based slice around line 121
// Method: <com.google.common.graph.StandardMutableValueGraph: boolean removeNode(N)>

  @Override
  @CanIgnoreReturnValue
  public @Nullable V putEdgeValue(EndpointPair<N> endpoints, V value) {
    validateEndpoints(endpoints);
    return putEdgeValue(endpoints.nodeU(), endpoints.nodeV(), value);
  }

  @Override
  @CanIgnoreReturnValue
  public boolean removeNode(N node) {
    checkNotNull(node, "node");

    GraphConnections<N, V> connections = nodeConnections.get(node);
    if (connections == null) {
      return false;
    }

    if (allowsSelfLoops()) {
      // Remove self-loop (if any) first, so we don't get CME while removing incident edges.
      if (connections.removeSuccessor(node) != null) {
