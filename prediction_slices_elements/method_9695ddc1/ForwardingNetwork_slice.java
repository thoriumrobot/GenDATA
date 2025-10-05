// Source-based slice around line 85
// Method: <com.google.common.graph.ForwardingNetwork: Set incidentEdges(N)>

    return delegate().predecessors(node);
  }

  @Override
  public Set<N> successors(N node) {
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
