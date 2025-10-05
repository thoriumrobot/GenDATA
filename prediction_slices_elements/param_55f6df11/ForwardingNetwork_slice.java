// Source-based slice around line 75
// Method: <com.google.common.graph.ForwardingNetwork: Set predecessors(N)>

    return delegate().edgeOrder();
  }

  @Override
  public Set<N> adjacentNodes(N node) {
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
