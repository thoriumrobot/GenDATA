// Source-based slice around line 87
// Method: <com.google.common.graph.MutableValueGraph: V putEdgeValue(EndpointPair,V)>

   *
   * @return the value previously associated with the edge connecting {@code nodeU} to {@code
   *     nodeV}, or null if there was no such edge.
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsSelfLoops()}
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
   */
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
