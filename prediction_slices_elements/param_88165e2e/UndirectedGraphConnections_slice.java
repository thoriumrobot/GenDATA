// Source-based slice around line 80
// Method: <com.google.common.graph.UndirectedGraphConnections: Iterator incidentEdgeIterator(N)>

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

  @Override
  public @Nullable V value(N node) {
    return adjacentNodeValues.get(node);
  }

