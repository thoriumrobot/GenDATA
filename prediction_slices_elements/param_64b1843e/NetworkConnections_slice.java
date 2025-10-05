// Source-based slice around line 76
// Method: <com.google.common.graph.NetworkConnections: void addOutEdge(E,N)>

  @CanIgnoreReturnValue
  N removeOutEdge(E edge);

  /**
   * Add {@code edge} to the set of incoming edges. Implicitly adds {@code node} as a predecessor.
   */
  void addInEdge(E edge, N node, boolean isSelfLoop);

  /** Add {@code edge} to the set of outgoing edges. Implicitly adds {@code node} as a successor. */
  void addOutEdge(E edge, N node);
}
