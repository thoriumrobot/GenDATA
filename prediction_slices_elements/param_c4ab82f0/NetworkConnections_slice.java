// Source-based slice around line 73
// Method: <com.google.common.graph.NetworkConnections: void addInEdge(E,N,boolean)>

  @Nullable N removeInEdge(E edge, boolean isSelfLoop);

  /** Remove {@code edge} from the set of outgoing edges. Returns the former successor node. */
  @CanIgnoreReturnValue
  N removeOutEdge(E edge);

  /**
   * Add {@code edge} to the set of incoming edges. Implicitly adds {@code node} as a predecessor.
   */
  void addInEdge(E edge, N node, boolean isSelfLoop);

  /** Add {@code edge} to the set of outgoing edges. Implicitly adds {@code node} as a successor. */
  void addOutEdge(E edge, N node);
}
