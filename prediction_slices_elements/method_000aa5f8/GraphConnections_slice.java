// Source-based slice around line 67
// Method: <com.google.common.graph.GraphConnections: void addPredecessor(N,V)>

   * the edge connecting the two nodes.
   */
  @CanIgnoreReturnValue
  @Nullable V removeSuccessor(N node);

  /**
   * Add {@code node} as a predecessor to the origin node. In the case of an undirected graph, it
   * also becomes a successor. Associates {@code value} with the edge connecting the two nodes.
   */
  void addPredecessor(N node, V value);

  /**
   * Add {@code node} as a successor to the origin node. In the case of an undirected graph, it also
   * becomes a predecessor. Associates {@code value} with the edge connecting the two nodes. Returns
   * the value previously associated with the edge connecting the two nodes.
   */
  @CanIgnoreReturnValue
  @Nullable V addSuccessor(N node, V value);
}
