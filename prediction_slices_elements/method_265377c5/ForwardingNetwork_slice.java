// Source-based slice around line 150
// Method: <com.google.common.graph.ForwardingNetwork: E edgeConnectingOrNull(EndpointPair)>

    return delegate().edgeConnecting(endpoints);
  }

  @Override
  public @Nullable E edgeConnectingOrNull(N nodeU, N nodeV) {
    return delegate().edgeConnectingOrNull(nodeU, nodeV);
  }

  @Override
  public @Nullable E edgeConnectingOrNull(EndpointPair<N> endpoints) {
    return delegate().edgeConnectingOrNull(endpoints);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
    return delegate().hasEdgeConnecting(nodeU, nodeV);
  }

  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
