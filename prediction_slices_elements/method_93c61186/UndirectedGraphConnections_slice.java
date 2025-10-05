// Source-based slice around line 70
// Method: <com.google.common.graph.UndirectedGraphConnections: Set predecessors()>

    return new UndirectedGraphConnections<>(ImmutableMap.copyOf(adjacentNodeValues));
  }

  @Override
  public Set<N> adjacentNodes() {
    return Collections.unmodifiableSet(adjacentNodeValues.keySet());
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
  public Iterator<EndpointPair<N>> incidentEdgeIterator(N thisNode) {
