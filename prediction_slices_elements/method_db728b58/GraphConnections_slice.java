// Source-based slice around line 75
// Method: <com.google.common.graph.GraphConnections: V addSuccessor(N,V)>

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
