// Source-based slice around line 98
// Method: <com.google.common.graph.UndirectedGraphConnections: V removeSuccessor(N)>

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
  public void addPredecessor(N node, V value) {
    @SuppressWarnings("unused")
    V unused = addSuccessor(node, value);
  }

  @Override
