// Source-based slice around line 106
// Method: <com.google.common.graph.StandardNetwork: boolean isDirected()>

    return nodeConnections.unmodifiableKeySet();
  }

  @Override
  public Set<E> edges() {
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
