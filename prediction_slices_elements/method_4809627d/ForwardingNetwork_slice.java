// Source-based slice around line 60
// Method: <com.google.common.graph.ForwardingNetwork: ElementOrder nodeOrder()>

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
    return delegate().edgeOrder();
  }

  @Override
  public Set<N> adjacentNodes(N node) {
