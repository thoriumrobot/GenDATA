// Source-based slice around line 140
// Method: <com.google.common.graph.ForwardingNetwork: Optional edgeConnecting(EndpointPair)>

    return delegate().edgesConnecting(endpoints);
  }

  @Override
  public Optional<E> edgeConnecting(N nodeU, N nodeV) {
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
