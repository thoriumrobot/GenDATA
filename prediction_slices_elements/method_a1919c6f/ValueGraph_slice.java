// Source-based slice around line 365
// Method: <com.google.common.graph.ValueGraph: V edgeValueOrDefault(EndpointPair,V)>

   * Returns the value of the edge that connects {@code endpoints} (in the order, if any, specified
   * by {@code endpoints}), if one is present; otherwise, returns {@code defaultValue}.
   *
   * <p>If this graph is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if either endpoint is not an element of this graph
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
   */
  @Nullable V edgeValueOrDefault(EndpointPair<N> endpoints, @Nullable V defaultValue);

  //
  // ValueGraph identity
  //

  /**
   * Returns {@code true} iff {@code object} is a {@link ValueGraph} that has the same elements and
   * the same structural relationships as those in this graph.
   *
   * <p>Thus, two value graphs A and B are equal if <b>all</b> of the following are true:
