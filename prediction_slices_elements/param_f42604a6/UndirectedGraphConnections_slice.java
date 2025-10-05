// Source-based slice around line 103
// Method: <com.google.common.graph.UndirectedGraphConnections: void addPredecessor(N,V)>

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
  public @Nullable V addSuccessor(N node, V value) {
    return adjacentNodeValues.put(node, value);
  }
}
