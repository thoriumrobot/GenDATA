// Source-based slice around line 96
// Method: <com.google.common.graph.MutableNetwork: boolean addEdge(EndpointPair,E)>

   * @return {@code true} if the network was modified as a result of this call
   * @throws IllegalArgumentException if {@code edge} already exists in the graph and connects some
   *     other endpoint pair that is not equal to {@code endpoints}
   * @throws IllegalArgumentException if the introduction of the edge would violate {@link
   *     #allowsParallelEdges()} or {@link #allowsSelfLoops()}
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
   */
  @CanIgnoreReturnValue
  boolean addEdge(EndpointPair<N> endpoints, E edge);

  /**
   * Removes {@code node} if it is present; all edges incident to {@code node} will also be removed.
   *
   * @return {@code true} if the network was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeNode(N node);

  /**
