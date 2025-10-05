// Source-based slice around line 98
// Method: <com.google.common.graph.EndpointPair: N nodeV()>

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

  /**
   * Returns the node that is adjacent to {@code node} along the origin edge.
   *
   * @throws IllegalArgumentException if this {@link EndpointPair} does not contain {@code node}
   * @since 20.0 (but the argument type was changed from {@code Object} to {@code N} in 31.0)
   */
  public final N adjacentNode(N node) {
