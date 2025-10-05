// Source-based slice around line 149
// Method: <com.google.common.graph.StandardValueGraph: V edgeValueOrDefault(EndpointPair,V)>

        && hasEdgeConnectingInternal(endpoints.nodeU(), endpoints.nodeV());
  }

  @Override
  public @Nullable V edgeValueOrDefault(N nodeU, N nodeV, @Nullable V defaultValue) {
    return edgeValueOrDefaultInternal(checkNotNull(nodeU), checkNotNull(nodeV), defaultValue);
  }

  @Override
  public @Nullable V edgeValueOrDefault(EndpointPair<N> endpoints, @Nullable V defaultValue) {
    validateEndpoints(endpoints);
    return edgeValueOrDefaultInternal(endpoints.nodeU(), endpoints.nodeV(), defaultValue);
  }

  @Override
  protected long edgeCount() {
    return edgeCount;
  }

  private final GraphConnections<N, V> checkedConnections(N node) {
