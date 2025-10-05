// Source-based slice around line 120
// Method: <com.google.common.graph.ForwardingNetwork: int outDegree(N)>

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
  public Set<E> edgesConnecting(N nodeU, N nodeV) {
    return delegate().edgesConnecting(nodeU, nodeV);
  }

  @Override
  public Set<E> edgesConnecting(EndpointPair<N> endpoints) {
