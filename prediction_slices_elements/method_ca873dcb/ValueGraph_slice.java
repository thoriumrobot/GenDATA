// Source-based slice around line 341
// Method: <com.google.common.graph.ValueGraph: Optional edgeValue(EndpointPair)>

   * Returns the value of the edge that connects {@code endpoints} (in the order, if any, specified
   * by {@code endpoints}), if one is present; otherwise, returns {@code Optional.empty()}.
   *
   * <p>If this graph is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if either endpoint is not an element of this graph
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
   */
  Optional<V> edgeValue(EndpointPair<N> endpoints);

  /**
   * Returns the value of the edge that connects {@code nodeU} to {@code nodeV}, if one is present;
   * otherwise, returns {@code defaultValue}.
   *
   * <p>In an undirected graph, this is equal to {@code edgeValueOrDefault(nodeV, nodeU,
   * defaultValue)}.
   *
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
   *     graph
