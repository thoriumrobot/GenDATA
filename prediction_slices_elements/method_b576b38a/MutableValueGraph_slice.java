// Source-based slice around line 104
// Method: <com.google.common.graph.MutableValueGraph: V removeEdge(N,N)>

  boolean removeNode(N node);

  /**
   * Removes the edge connecting {@code nodeU} to {@code nodeV}, if it is present.
   *
   * @return the value previously associated with the edge connecting {@code nodeU} to {@code
   *     nodeV}, or null if there was no such edge.
   */
  @CanIgnoreReturnValue
  @Nullable V removeEdge(N nodeU, N nodeV);

  /**
   * Removes the edge connecting {@code endpoints}, if it is present.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
   * @return the value previously associated with the edge connecting {@code endpoints}, or null if
   *     there was no such edge.
   * @since 27.1
   */
