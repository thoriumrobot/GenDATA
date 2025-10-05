// Source-based slice around line 64
// Method: <com.google.common.graph.NetworkConnections: N removeInEdge(E,boolean)>

   */
  N adjacentNode(E edge);

  /**
   * Remove {@code edge} from the set of incoming edges. Returns the former predecessor node.
   *
   * <p>In the undirected case, returns {@code null} if {@code isSelfLoop} is true.
   */
  @CanIgnoreReturnValue
  @Nullable N removeInEdge(E edge, boolean isSelfLoop);

  /** Remove {@code edge} from the set of outgoing edges. Returns the former successor node. */
  @CanIgnoreReturnValue
  N removeOutEdge(E edge);

  /**
   * Add {@code edge} to the set of incoming edges. Implicitly adds {@code node} as a predecessor.
   */
  void addInEdge(E edge, N node, boolean isSelfLoop);

