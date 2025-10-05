// Source-based slice around line 95
// Method: <com.google.common.graph.ForwardingNetwork: Set outEdges(N)>

    return delegate().incidentEdges(node);
  }

  @Override
  public Set<E> inEdges(N node) {
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
