// Source-based slice around line 144
// Method: <com.google.common.graph.StandardValueGraph: V edgeValueOrDefault(N,N,V)>


  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
    checkNotNull(endpoints);
    return isOrderingCompatible(endpoints)
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
