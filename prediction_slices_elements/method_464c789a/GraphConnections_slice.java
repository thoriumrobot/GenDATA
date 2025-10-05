// Source-based slice around line 61
// Method: <com.google.common.graph.GraphConnections: V removeSuccessor(N)>


  /** Remove {@code node} from the set of predecessors. */
  void removePredecessor(N node);

  /**
   * Remove {@code node} from the set of successors. Returns the value previously associated with
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
