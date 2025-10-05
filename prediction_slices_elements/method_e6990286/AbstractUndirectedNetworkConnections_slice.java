// Source-based slice around line 97
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: void addOutEdge(E,N)>


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
}
