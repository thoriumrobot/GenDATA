// Source-based slice around line 90
// Method: <com.google.common.graph.EndpointPair: N nodeU()>

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
   * Returns the node {@link #adjacentNode(Object) adjacent} to {@link #nodeU()} along the origin
   * edge. If this {@link EndpointPair} {@link #isOrdered()}, this is equal to {@link #target()}.
   */
  public final N nodeV() {
    return nodeV;
  }
