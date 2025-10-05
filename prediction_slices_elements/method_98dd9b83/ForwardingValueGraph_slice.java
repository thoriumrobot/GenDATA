// Source-based slice around line 109
// Method: <com.google.common.graph.ForwardingValueGraph: Optional edgeValue(N,N)>

    return delegate().hasEdgeConnecting(nodeU, nodeV);
  }

  @Override
  public boolean hasEdgeConnecting(EndpointPair<N> endpoints) {
    return delegate().hasEdgeConnecting(endpoints);
  }

  @Override
  public Optional<V> edgeValue(N nodeU, N nodeV) {
    return delegate().edgeValue(nodeU, nodeV);
  }

  @Override
  public Optional<V> edgeValue(EndpointPair<N> endpoints) {
    return delegate().edgeValue(endpoints);
  }

  @Override
  public @Nullable V edgeValueOrDefault(N nodeU, N nodeV, @Nullable V defaultValue) {
