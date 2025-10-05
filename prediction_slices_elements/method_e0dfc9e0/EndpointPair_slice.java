// Source-based slice around line 84
// Method: <com.google.common.graph.EndpointPair: N target()>

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
   * returns an arbitrary (but consistent) endpoint of the origin edge.
   */
  public final N nodeU() {
    return nodeU;
  }

  /**
