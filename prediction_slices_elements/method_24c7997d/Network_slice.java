// Source-based slice around line 498
// Method: <com.google.common.graph.Network: boolean hasEdgeConnecting(EndpointPair)>

   * any, specified by {@code endpoints}).
   *
   * <p>Unlike the other {@code EndpointPair}-accepting methods, this method does not throw if the
   * endpoints are unordered and the network is directed; it simply returns {@code false}. This is
   * for consistency with {@link Graph#hasEdgeConnecting(EndpointPair)} and {@link
   * ValueGraph#hasEdgeConnecting(EndpointPair)}.
   *
   * @since 27.1
   */
  boolean hasEdgeConnecting(EndpointPair<N> endpoints);

  //
  // Network identity
  //

  /**
   * Returns {@code true} iff {@code object} is a {@link Network} that has the same elements and the
   * same structural relationships as those in this network.
   *
   * <p>Thus, two networks A and B are equal if <b>all</b> of the following are true:
