// Source-based slice around line 90
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: void addInEdge(E,N,boolean)>


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
  }

  @Override
  public void addOutEdge(E edge, N node) {
    N previousNode = incidentEdgeMap.put(edge, node);
    checkState(previousNode == null);
  }
