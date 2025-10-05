// Source-based slice around line 89
// Method: <com.google.common.graph.StandardValueGraph: boolean isDirected()>

    this.edgeCount = checkNonNegative(edgeCount);
  }

  @Override
  public Set<N> nodes() {
    return nodeConnections.unmodifiableKeySet();
  }

  @Override
  public boolean isDirected() {
    return isDirected;
  }

  @Override
  public boolean allowsSelfLoops() {
    return allowsSelfLoops;
  }

  @Override
  public ElementOrder<N> nodeOrder() {
