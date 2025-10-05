// Source-based slice around line 223
// Method: <com.google.common.graph.BaseGraph: boolean hasEdgeConnecting(EndpointPair)>

   *
   * <p>Unlike the other {@code EndpointPair}-accepting methods, this method does not throw if the
   * endpoints are unordered; it simply returns false. This is for consistency with the behavior of
   * {@link Collection#contains(Object)} (which does not generally throw if the object cannot be
   * present in the collection), and the desire to have this method's behavior be compatible with
   * {@code edges().contains(endpoints)}.
   *
   * @since 27.1
   */
  boolean hasEdgeConnecting(EndpointPair<N> endpoints);
}
