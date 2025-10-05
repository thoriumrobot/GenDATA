// Source-based slice around line 83
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: N removeOutEdge(E)>

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
    return requireNonNull(previousNode);
  }

  @Override
  public void addInEdge(E edge, N node, boolean isSelfLoop) {
    if (!isSelfLoop) {
      addOutEdge(edge, node);
    }
