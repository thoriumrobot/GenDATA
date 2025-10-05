// Source-based slice around line 104
// Method: <com.google.common.graph.ForwardingValueGraph: boolean hasEdgeConnecting(EndpointPair)>

    return delegate().outDegree(node);
  }

  @Override
  public boolean hasEdgeConnecting(N nodeU, N nodeV) {
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
