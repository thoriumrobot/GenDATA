// Source-based slice around line 417
// Method: <com.google.common.graph.Network: Set edgesConnecting(EndpointPair)>

   *   <li>{@code hashCode()} does not throw
   *   <li>if either endpoint is re-added to the network after having been removed, {@code view}'s
   *       behavior is undefined
   * </ul>
   *
   * @throws IllegalArgumentException if either endpoint is not an element of this network
   * @throws IllegalArgumentException if the endpoints are unordered and the network is directed
   * @since 27.1
   */
  Set<E> edgesConnecting(EndpointPair<N> endpoints);

  /**
   * Returns the single edge that directly connects {@code nodeU} to {@code nodeV}, if one is
   * present, or {@code Optional.empty()} if no such edge exists.
   *
   * <p>In an undirected network, this is equal to {@code edgeConnecting(nodeV, nodeU)}.
   *
   * @throws IllegalArgumentException if there are multiple parallel edges connecting {@code nodeU}
   *     to {@code nodeV}
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
