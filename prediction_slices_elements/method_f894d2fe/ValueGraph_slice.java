// Source-based slice around line 318
// Method: <com.google.common.graph.ValueGraph: boolean hasEdgeConnecting(EndpointPair)>

   * <p>Unlike the other {@code EndpointPair}-accepting methods, this method does not throw if the
   * endpoints are unordered and the graph is directed; it simply returns {@code false}. This is for
   * consistency with the behavior of {@link Collection#contains(Object)} (which does not generally
   * throw if the object cannot be present in the collection), and the desire to have this method's
   * behavior be compatible with {@code edges().contains(endpoints)}.
   *
   * @since 27.1
   */
  @Override
  boolean hasEdgeConnecting(EndpointPair<N> endpoints);

  /**
   * Returns the value of the edge that connects {@code nodeU} to {@code nodeV} (in the order, if
   * any, specified by {@code endpoints}), if one is present; otherwise, returns {@code
   * Optional.empty()}.
   *
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
   *     graph
   * @since 23.0 (since 20.0 with return type {@code V})
   */
