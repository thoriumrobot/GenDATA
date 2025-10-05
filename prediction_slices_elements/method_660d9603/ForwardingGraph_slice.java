// Source-based slice around line 56
// Method: <com.google.common.graph.ForwardingGraph: ElementOrder nodeOrder()>

    return delegate().isDirected();
  }

  @Override
  public boolean allowsSelfLoops() {
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
