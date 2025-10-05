// Source-based slice around line 117
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: N removeOutEdge(E)>

    if (isSelfLoop) {
      checkNonNegative(--selfLoopCount);
    }
    N previousNode = inEdgeMap.remove(edge);
    // We're relying on callers to call this method only with an edge that's in the graph.
    return requireNonNull(previousNode);
  }

  @Override
  public N removeOutEdge(E edge) {
    N previousNode = outEdgeMap.remove(edge);
    // We're relying on callers to call this method only with an edge that's in the graph.
    return requireNonNull(previousNode);
  }

  @Override
  public void addInEdge(E edge, N node, boolean isSelfLoop) {
    checkNotNull(edge);
    checkNotNull(node);

