// Source-based slice around line 92
// Method: <com.google.common.graph.UndirectedGraphConnections: void removePredecessor(N)>

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
  public @Nullable V removeSuccessor(N node) {
    return adjacentNodeValues.remove(node);
  }

  @Override
