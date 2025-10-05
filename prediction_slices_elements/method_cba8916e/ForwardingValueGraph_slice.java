// Source-based slice around line 94
// Method: <com.google.common.graph.ForwardingValueGraph: int outDegree(N)>

    return delegate().degree(node);
  }

  @Override
  public int inDegree(N node) {
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
