// Source-based slice around line 125
// Method: <com.google.common.graph.ForwardingNetwork: Set edgesConnecting(N,N)>

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
    return delegate().edgesConnecting(endpoints);
  }

  @Override
  public Optional<E> edgeConnecting(N nodeU, N nodeV) {
