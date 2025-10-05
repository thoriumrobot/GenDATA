// Source-based slice around line 75
// Method: <com.google.common.graph.UndirectedGraphConnections: Set successors()>

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
    return Iterators.transform(
        adjacentNodeValues.keySet().iterator(),
        (N incidentNode) -> EndpointPair.unordered(thisNode, incidentNode));
  }

