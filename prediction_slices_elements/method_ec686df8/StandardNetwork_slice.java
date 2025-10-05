// Source-based slice around line 111
// Method: <com.google.common.graph.StandardNetwork: boolean allowsParallelEdges()>

    return edgeToReferenceNode.unmodifiableKeySet();
  }

  @Override
  public boolean isDirected() {
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
