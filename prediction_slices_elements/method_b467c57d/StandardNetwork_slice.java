// Source-based slice around line 116
// Method: <com.google.common.graph.StandardNetwork: boolean allowsSelfLoops()>

    return isDirected;
  }

  @Override
  public boolean allowsParallelEdges() {
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
