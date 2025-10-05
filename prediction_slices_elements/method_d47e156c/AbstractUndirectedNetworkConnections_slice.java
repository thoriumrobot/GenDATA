// Source-based slice around line 54
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: Set incidentEdges()>

    return adjacentNodes();
  }

  @Override
  public Set<N> successors() {
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
