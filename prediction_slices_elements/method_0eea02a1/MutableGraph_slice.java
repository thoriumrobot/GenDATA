// Source-based slice around line 79
// Method: <com.google.common.graph.MutableGraph: boolean putEdge(EndpointPair)>

   * {@link #addNode(Object) add} each missing endpoint to the graph.
   *
   * @return {@code true} if the graph was modified as a result of this call
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsSelfLoops()}
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
   */
  @CanIgnoreReturnValue
  boolean putEdge(EndpointPair<N> endpoints);

  /**
   * Removes {@code node} if it is present; all edges incident to {@code node} will also be removed.
   *
   * @return {@code true} if the graph was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeNode(N node);

  /**
