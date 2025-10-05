// Source-based slice around line 474
// Method: <com.google.common.graph.Network: E edgeConnectingOrNull(EndpointPair)>

   *
   * <p>If this network is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if either endpoint is not an element of this network
   * @throws IllegalArgumentException if the endpoints are unordered and the network is directed
   * @since 27.1
   */
  @Nullable E edgeConnectingOrNull(EndpointPair<N> endpoints);

  /**
   * Returns true if there is an edge that directly connects {@code nodeU} to {@code nodeV}. This is
   * equivalent to {@code nodes().contains(nodeU) && successors(nodeU).contains(nodeV)}, and to
   * {@code edgeConnectingOrNull(nodeU, nodeV) != null}.
   *
   * <p>In an undirected network, this is equal to {@code hasEdgeConnecting(nodeV, nodeU)}.
   *
   * @since 23.0
   */
