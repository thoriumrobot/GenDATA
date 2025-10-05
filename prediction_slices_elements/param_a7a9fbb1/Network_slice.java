// Source-based slice around line 525
// Method: <com.google.common.graph.Network: boolean equals(Object)>

   *
   * <p>Network properties besides {@link #isDirected() directedness} do <b>not</b> affect equality.
   * For example, two networks may be considered equal even if one allows parallel edges and the
   * other doesn't. Additionally, the order in which nodes or edges are added to the network, and
   * the order in which they are iterated over, are irrelevant.
   *
   * <p>A reference implementation of this is provided by {@link AbstractNetwork#equals(Object)}.
   */
  @Override
  boolean equals(@Nullable Object object);

  /**
   * Returns the hash code for this network. The hash code of a network is defined as the hash code
   * of a map from each of its {@link #edges() edges} to their {@link #incidentNodes(Object)
   * incident nodes}.
   *
   * <p>A reference implementation of this is provided by {@link AbstractNetwork#hashCode()}.
   */
  @Override
  int hashCode();
