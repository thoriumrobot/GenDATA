// Source-based slice around line 50
// Method: <com.google.common.graph.ForwardingNetwork: boolean allowsParallelEdges()>

    return delegate().edges();
  }

  @Override
  public boolean isDirected() {
    return delegate().isDirected();
  }

  @Override
  public boolean allowsParallelEdges() {
    return delegate().allowsParallelEdges();
  }

  @Override
  public boolean allowsSelfLoops() {
    return delegate().allowsSelfLoops();
  }

  @Override
  public ElementOrder<N> nodeOrder() {
