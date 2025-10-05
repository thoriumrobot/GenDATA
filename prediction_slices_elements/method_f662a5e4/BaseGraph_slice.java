// Source-based slice around line 208
// Method: <com.google.common.graph.BaseGraph: boolean hasEdgeConnecting(N,N)>


  /**
   * Returns true if there is an edge that directly connects {@code nodeU} to {@code nodeV}. This is
   * equivalent to {@code nodes().contains(nodeU) && successors(nodeU).contains(nodeV)}.
   *
   * <p>In an undirected graph, this is equal to {@code hasEdgeConnecting(nodeV, nodeU)}.
   *
   * @since 23.0
   */
  boolean hasEdgeConnecting(N nodeU, N nodeV);

  /**
   * Returns true if there is an edge that directly connects {@code endpoints} (in the order, if
   * any, specified by {@code endpoints}). This is equivalent to {@code
   * edges().contains(endpoints)}.
   *
   * <p>Unlike the other {@code EndpointPair}-accepting methods, this method does not throw if the
   * endpoints are unordered; it simply returns false. This is for consistency with the behavior of
   * {@link Collection#contains(Object)} (which does not generally throw if the object cannot be
   * present in the collection), and the desire to have this method's behavior be compatible with
