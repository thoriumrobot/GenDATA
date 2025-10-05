// Source-based slice around line 145
// Method: <com.google.common.graph.ForwardingNetwork: E edgeConnectingOrNull(N,N)>

    return delegate().edgeConnecting(nodeU, nodeV);
  }

  @Override
  public Optional<E> edgeConnecting(EndpointPair<N> endpoints) {
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
