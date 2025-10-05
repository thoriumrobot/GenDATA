// Source-based slice around line 392
// Method: <com.google.common.graph.ValueGraph: boolean equals(Object)>

   *
   * <p>Graph properties besides {@link #isDirected() directedness} do <b>not</b> affect equality.
   * For example, two graphs may be considered equal even if one allows self-loops and the other
   * doesn't. Additionally, the order in which nodes or edges are added to the graph, and the order
   * in which they are iterated over, are irrelevant.
   *
   * <p>A reference implementation of this is provided by {@link AbstractValueGraph#equals(Object)}.
   */
  @Override
  boolean equals(@Nullable Object object);

  /**
   * Returns the hash code for this graph. The hash code of a graph is defined as the hash code of a
   * map from each of its {@link #edges() edges} to the associated {@link #edgeValue(N, N) edge
   * value}.
   *
   * <p>A reference implementation of this is provided by {@link AbstractValueGraph#hashCode()}.
   */
  @Override
  int hashCode();
