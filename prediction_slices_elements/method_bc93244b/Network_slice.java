// Source-based slice around line 446
// Method: <com.google.common.graph.Network: Optional edgeConnecting(EndpointPair)>

   *
   * <p>If this network is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if either endpoint is not an element of this network
   * @throws IllegalArgumentException if the endpoints are unordered and the network is directed
   * @since 27.1
   */
  Optional<E> edgeConnecting(EndpointPair<N> endpoints);

  /**
   * Returns the single edge that directly connects {@code nodeU} to {@code nodeV}, if one is
   * present, or {@code null} if no such edge exists.
   *
   * <p>In an undirected network, this is equal to {@code edgeConnectingOrNull(nodeV, nodeU)}.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
