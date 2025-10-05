// Source-based slice around line 100
// Method: <com.google.common.graph.ForwardingNetwork: EndpointPair incidentNodes(E)>

    return delegate().inEdges(node);
  }

  @Override
  public Set<E> outEdges(N node) {
    return delegate().outEdges(node);
  }

  @Override
  public EndpointPair<N> incidentNodes(E edge) {
    return delegate().incidentNodes(edge);
  }

  @Override
  public Set<E> adjacentEdges(E edge) {
    return delegate().adjacentEdges(edge);
  }

  @Override
  public int degree(N node) {
