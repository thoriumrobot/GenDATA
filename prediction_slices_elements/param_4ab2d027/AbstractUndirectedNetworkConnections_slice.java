// Source-based slice around line 69
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: N adjacentNode(E)>

    return incidentEdges();
  }

  @Override
  public Set<E> outEdges() {
    return incidentEdges();
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
