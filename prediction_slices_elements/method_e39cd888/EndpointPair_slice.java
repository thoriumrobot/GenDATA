// Source-based slice around line 77
// Method: <com.google.common.graph.EndpointPair: N source()>

  static <N> EndpointPair<N> of(Network<?, ?> network, N nodeU, N nodeV) {
    return network.isDirected() ? ordered(nodeU, nodeV) : unordered(nodeU, nodeV);
  }

  /**
   * If this {@link EndpointPair} {@link #isOrdered()}, returns the node which is the source.
   *
   * @throws UnsupportedOperationException if this {@link EndpointPair} is not ordered
   */
  public abstract N source();

  /**
   * If this {@link EndpointPair} {@link #isOrdered()}, returns the node which is the target.
   *
   * @throws UnsupportedOperationException if this {@link EndpointPair} is not ordered
   */
  public abstract N target();

  /**
   * If this {@link EndpointPair} {@link #isOrdered()} returns the {@link #source()}; otherwise,
