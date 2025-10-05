// Source-based slice around line 90
// Method: <com.google.common.graph.ForwardingNetwork: Set inEdges(N)>

    return delegate().successors(node);
  }

  @Override
  public Set<E> incidentEdges(N node) {
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
