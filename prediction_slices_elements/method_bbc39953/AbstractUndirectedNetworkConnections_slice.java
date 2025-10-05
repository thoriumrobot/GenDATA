// Source-based slice around line 75
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: N removeInEdge(E,boolean)>

  }

  @Override
  public N adjacentNode(E edge) {
    // We're relying on callers to call this method only with an edge that's in the graph.
    return requireNonNull(incidentEdgeMap.get(edge));
  }

  @Override
  public @Nullable N removeInEdge(E edge, boolean isSelfLoop) {
    if (!isSelfLoop) {
      return removeOutEdge(edge);
    }
    return null;
  }

  @Override
  public N removeOutEdge(E edge) {
    N previousNode = incidentEdgeMap.remove(edge);
    // We're relying on callers to call this method only with an edge that's in the graph.
