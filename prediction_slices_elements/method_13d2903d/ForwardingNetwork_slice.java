// Source-based slice around line 130
// Method: <com.google.common.graph.ForwardingNetwork: Set edgesConnecting(EndpointPair)>

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
    return delegate().edgeConnecting(nodeU, nodeV);
  }

  @Override
  public Optional<E> edgeConnecting(EndpointPair<N> endpoints) {
