// Source-based slice around line 124
// Method: <com.google.common.graph.AbstractDirectedNetworkConnections: void addInEdge(E,N,boolean)>


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

    if (isSelfLoop) {
      checkPositive(++selfLoopCount);
    }
    N previousNode = inEdgeMap.put(edge, node);
    checkState(previousNode == null);
  }

