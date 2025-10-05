// Source-based slice around line 51
// Method: <com.google.common.graph.GraphConnections: V value(N)>

   *
   * @param thisNode The node that this all of the connections in this class are connected to.
   */
  Iterator<EndpointPair<N>> incidentEdgeIterator(N thisNode);

  /**
   * Returns the value associated with the edge connecting the origin node to {@code node}, or null
   * if there is no such edge.
   */
  @Nullable V value(N node);

  /** Remove {@code node} from the set of predecessors. */
  void removePredecessor(N node);

  /**
   * Remove {@code node} from the set of successors. Returns the value previously associated with
   * the edge connecting the two nodes.
   */
  @CanIgnoreReturnValue
  @Nullable V removeSuccessor(N node);
