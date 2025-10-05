// Source-based slice around line 109
// Method: <com.google.common.graph.UndirectedGraphConnections: V addSuccessor(N,V)>

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
