// Source-based slice around line 87
// Method: <com.google.common.graph.UndirectedGraphConnections: V value(N)>


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

  @Override
  public void removePredecessor(N node) {
    @SuppressWarnings("unused")
    V unused = removeSuccessor(node);
  }

  @Override
