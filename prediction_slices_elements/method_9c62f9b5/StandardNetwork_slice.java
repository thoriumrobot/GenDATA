// Source-based slice around line 121
// Method: <com.google.common.graph.StandardNetwork: ElementOrder nodeOrder()>

    return allowsParallelEdges;
  }

  @Override
  public boolean allowsSelfLoops() {
    return allowsSelfLoops;
  }

  @Override
  public ElementOrder<N> nodeOrder() {
    return nodeOrder;
  }

  @Override
  public ElementOrder<E> edgeOrder() {
    return edgeOrder;
  }

  @Override
  public Set<E> incidentEdges(N node) {
