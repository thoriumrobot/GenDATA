// Source-based slice around line 49
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: Set successors()>

    this.incidentEdgeMap = checkNotNull(incidentEdgeMap);
  }

  @Override
  public Set<N> predecessors() {
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
