// Source-based slice around line 114
// Method: <com.google.common.graph.ForwardingValueGraph: Optional edgeValue(EndpointPair)>

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
    return delegate().edgeValueOrDefault(nodeU, nodeV, defaultValue);
  }

  @Override
  public @Nullable V edgeValueOrDefault(EndpointPair<N> endpoints, @Nullable V defaultValue) {
