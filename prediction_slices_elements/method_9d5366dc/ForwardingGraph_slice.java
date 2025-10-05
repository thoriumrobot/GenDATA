// Source-based slice around line 61
// Method: <com.google.common.graph.ForwardingGraph: ElementOrder incidentEdgeOrder()>

    return delegate().allowsSelfLoops();
  }

  @Override
  public ElementOrder<N> nodeOrder() {
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
