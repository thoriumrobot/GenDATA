// Source-based slice around line 135
// Method: <com.google.common.graph.ForwardingNetwork: Optional edgeConnecting(N,N)>

    return delegate().edgesConnecting(nodeU, nodeV);
  }

  @Override
  public Set<E> edgesConnecting(EndpointPair<N> endpoints) {
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
