// Source-based slice around line 87
// Method: <com.google.common.graph.MutableGraph: boolean removeNode(N)>

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
   * Removes the edge connecting {@code nodeU} to {@code nodeV}, if it is present.
   *
   * @return {@code true} if the graph was modified as a result of this call
   */
  @CanIgnoreReturnValue
  boolean removeEdge(N nodeU, N nodeV);

  /**
