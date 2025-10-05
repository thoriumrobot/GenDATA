// Source-based slice around line 59
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: Set inEdges()>

    return adjacentNodes();
  }

  @Override
  public Set<E> incidentEdges() {
    return Collections.unmodifiableSet(incidentEdgeMap.keySet());
  }

  @Override
  public Set<E> inEdges() {
    return incidentEdges();
  }

  @Override
  public Set<E> outEdges() {
    return incidentEdges();
  }

  @Override
  public N adjacentNode(E edge) {
