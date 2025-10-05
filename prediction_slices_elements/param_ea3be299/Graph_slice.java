// Source-based slice around line 315
// Method: <com.google.common.graph.Graph: boolean hasEdgeConnecting(EndpointPair)>

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

  //
  // Graph identity
  //

  /**
   * Returns {@code true} iff {@code object} is a {@link Graph} that has the same elements and the
   * same structural relationships as those in this graph.
   *
   * <p>Thus, two graphs A and B are equal if <b>all</b> of the following are true:
