// Source-based slice around line 69
// Method: <com.google.common.graph.ForwardingValueGraph: Set adjacentNodes(N)>

    return delegate().nodeOrder();
  }

  @Override
  public ElementOrder<N> incidentEdgeOrder() {
    return delegate().incidentEdgeOrder();
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
