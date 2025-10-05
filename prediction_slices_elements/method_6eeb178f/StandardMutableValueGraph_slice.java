// Source-based slice around line 114
// Method: <com.google.common.graph.StandardMutableValueGraph: V putEdgeValue(EndpointPair,V)>

    connectionsV.addPredecessor(nodeU, value);
    if (previousValue == null) {
      checkPositive(++edgeCount);
    }
    return previousValue;
  }

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
