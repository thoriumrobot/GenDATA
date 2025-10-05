// Source-based slice around line 56
// Method: <com.google.common.graph.NetworkConnections: N adjacentNode(E)>

   * parallel edges, this set cannot be of size greater than one.
   */
  Set<E> edgesConnecting(N node);

  /**
   * Returns the node that is adjacent to the origin node along {@code edge}.
   *
   * <p>In the directed case, {@code edge} is assumed to be an outgoing edge.
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
