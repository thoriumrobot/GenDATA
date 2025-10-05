// Source-based slice around line 65
// Method: <com.google.common.graph.ForwardingNetwork: ElementOrder edgeOrder()>

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
    return delegate().adjacentNodes(node);
  }

  @Override
  public Set<N> predecessors(N node) {
