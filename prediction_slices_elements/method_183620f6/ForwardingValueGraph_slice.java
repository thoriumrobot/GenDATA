// Source-based slice around line 54
// Method: <com.google.common.graph.ForwardingValueGraph: boolean allowsSelfLoops()>

    return delegate().edges().size();
  }

  @Override
  public boolean isDirected() {
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
