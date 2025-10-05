// Source-based slice around line 105
// Method: <com.google.common.graph.ForwardingNetwork: Set adjacentEdges(E)>

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
    return delegate().degree(node);
  }

  @Override
  public int inDegree(N node) {
