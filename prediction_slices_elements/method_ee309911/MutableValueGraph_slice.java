// Source-based slice around line 95
// Method: <com.google.common.graph.MutableValueGraph: boolean removeNode(N)>

  @CanIgnoreReturnValue
  @Nullable V putEdgeValue(EndpointPair<N> endpoints, V value);

  /**
   * Removes {@code node} if it is present; all edges incident to {@code node} will also be removed.
   *
   * @return {@code true} if the graph was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeNode(N node);

  /**
   * Removes the edge connecting {@code nodeU} to {@code nodeV}, if it is present.
   *
   * @return the value previously associated with the edge connecting {@code nodeU} to {@code
   *     nodeV}, or null if there was no such edge.
   */
  @CanIgnoreReturnValue
  @Nullable V removeEdge(N nodeU, N nodeV);

