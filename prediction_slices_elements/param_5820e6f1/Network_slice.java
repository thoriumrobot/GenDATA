// Source-based slice around line 485
// Method: <com.google.common.graph.Network: boolean hasEdgeConnecting(N,N)>

  /**
   * Returns true if there is an edge that directly connects {@code nodeU} to {@code nodeV}. This is
   * equivalent to {@code nodes().contains(nodeU) && successors(nodeU).contains(nodeV)}, and to
   * {@code edgeConnectingOrNull(nodeU, nodeV) != null}.
   *
   * <p>In an undirected network, this is equal to {@code hasEdgeConnecting(nodeV, nodeU)}.
   *
   * @since 23.0
   */
  boolean hasEdgeConnecting(N nodeU, N nodeV);

  /**
   * Returns true if there is an edge that directly connects {@code endpoints} (in the order, if
   * any, specified by {@code endpoints}).
   *
   * <p>Unlike the other {@code EndpointPair}-accepting methods, this method does not throw if the
   * endpoints are unordered and the network is directed; it simply returns {@code false}. This is
   * for consistency with {@link Graph#hasEdgeConnecting(EndpointPair)} and {@link
   * ValueGraph#hasEdgeConnecting(EndpointPair)}.
   *
