// Source-based slice around line 329
// Method: <com.google.common.graph.ValueGraph: Optional edgeValue(N,N)>

  /**
   * Returns the value of the edge that connects {@code nodeU} to {@code nodeV} (in the order, if
   * any, specified by {@code endpoints}), if one is present; otherwise, returns {@code
   * Optional.empty()}.
   *
   * @throws IllegalArgumentException if {@code nodeU} or {@code nodeV} is not an element of this
   *     graph
   * @since 23.0 (since 20.0 with return type {@code V})
   */
  Optional<V> edgeValue(N nodeU, N nodeV);

  /**
   * Returns the value of the edge that connects {@code endpoints} (in the order, if any, specified
   * by {@code endpoints}), if one is present; otherwise, returns {@code Optional.empty()}.
   *
   * <p>If this graph is directed, the endpoints must be ordered.
   *
   * @throws IllegalArgumentException if either endpoint is not an element of this graph
   * @throws IllegalArgumentException if the endpoints are unordered and the graph is directed
   * @since 27.1
