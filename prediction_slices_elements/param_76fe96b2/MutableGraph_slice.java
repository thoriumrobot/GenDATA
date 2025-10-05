// Source-based slice around line 107
// Method: <com.google.common.graph.MutableGraph: boolean removeEdge(EndpointPair)>

   * Removes the edge connecting {@code endpoints}, if it is present.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @return {@code true} if the graph was modified as a result of this call
   * @since 27.1
   */
  @CanIgnoreReturnValue
  boolean removeEdge(EndpointPair<N> endpoints);
}
