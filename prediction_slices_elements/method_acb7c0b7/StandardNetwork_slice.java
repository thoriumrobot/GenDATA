// Source-based slice around line 101
// Method: <com.google.common.graph.StandardNetwork: Set edges()>

    this.edgeToReferenceNode = new MapIteratorCache<>(edgeToReferenceNode);
  }

  @Override
  public Set<N> nodes() {
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
