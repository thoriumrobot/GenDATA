// Source-based slice around line 460
// Method: <com.google.common.graph.Network: E edgeConnectingOrNull(N,N)>

   *
   * <p>In an undirected network, this is equal to {@code edgeConnectingOrNull(nodeV, nodeU)}.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
   *     network
   * @since 23.0
   */
  @Nullable E edgeConnectingOrNull(N nodeU, N nodeV);

  /**
   * Returns the single edge that directly connects {@code endpoints} (in the order, if any,
   * specified by {@code endpoints}), if one is present, or {@code null} if no such edge exists.
   *
   * <p>If this network is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if either endpoint is not an element of this network
