// Source-based slice around line 55
// Method: <com.google.common.graph.ForwardingNetwork: boolean allowsSelfLoops()>

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
    return delegate().nodeOrder();
  }

  @Override
  public ElementOrder<E> edgeOrder() {
