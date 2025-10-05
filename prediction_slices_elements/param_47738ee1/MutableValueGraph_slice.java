// Source-based slice around line 116
// Method: <com.google.common.graph.MutableValueGraph: V removeEdge(EndpointPair)>

   * Removes the edge connecting {@code endpoints}, if it is present.
   *
   * <p>If this graph is directed, {@code endpoints} must be ordered.
   *
   * @return the value previously associated with the edge connecting {@code endpoints}, or null if
   *     there was no such edge.
   * @since 27.1
   */
  @CanIgnoreReturnValue
  @Nullable V removeEdge(EndpointPair<N> endpoints);
}
