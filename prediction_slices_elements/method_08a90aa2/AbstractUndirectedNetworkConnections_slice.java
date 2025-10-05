// Source-based slice around line 64
// Method: <com.google.common.graph.AbstractUndirectedNetworkConnections: Set outEdges()>

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
    // We're relying on callers to call this method only with an edge that's in the graph.
    return requireNonNull(incidentEdgeMap.get(edge));
  }

  @Override
