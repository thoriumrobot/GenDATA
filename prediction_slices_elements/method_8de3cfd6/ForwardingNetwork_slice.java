// Source-based slice around line 80
// Method: <com.google.common.graph.ForwardingNetwork: Set successors(N)>

    return delegate().adjacentNodes(node);
  }

  @Override
  public Set<N> predecessors(N node) {
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
