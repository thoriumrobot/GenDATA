// Source-based slice around line 99
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: N adjacentNode(E)>

    return Collections.unmodifiableSet(inEdgeMap.keySet());
  }

  @Override
  public Set<E> outEdges() {
    return Collections.unmodifiableSet(outEdgeMap.keySet());
  }

  @Override
  public N adjacentNode(E edge) {
    // Since the reference node is defined to be 'source' for directed graphs,
    // we can assume this edge lives in the set of outgoing edges.
    // (We're relying on callers to call this method only with an edge that's in the graph.)
    return requireNonNull(outEdgeMap.get(edge));
  }

  @Override
  public N removeInEdge(E edge, boolean isSelfLoop) {
    if (isSelfLoop) {
      checkNonNegative(--selfLoopCount);
