// Source-based slice around line 95
// Method: <com.google.common.graph.MutableGraph: boolean removeEdge(N,N)>

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
   * Removes the edge connecting {@code endpoints}, if it is present.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @return {@code true} if the graph was modified as a result of this call
   * @since 27.1
   */
