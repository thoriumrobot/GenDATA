// Source-based slice around line 94
// Method: <com.google.common.graph.StandardValueGraph: boolean allowsSelfLoops()>

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
    return nodeOrder;
  }

  @Override
  public Set<N> adjacentNodes(N node) {
