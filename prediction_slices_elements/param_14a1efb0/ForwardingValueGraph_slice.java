// Source-based slice around line 99
// Method: <com.google.common.graph.ForwardingValueGraph: boolean hasEdgeConnecting(N,N)>

    return delegate().inDegree(node);
  }

  @Override
  public int outDegree(N node) {
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
