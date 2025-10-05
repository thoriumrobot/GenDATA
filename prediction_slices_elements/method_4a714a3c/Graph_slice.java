// Source-based slice around line 350
// Method: <com.google.common.graph.Graph: int hashCode()>

  boolean equals(@Nullable Object object);

  /**
   * Returns the hash code for this graph. The hash code of a graph is defined as the hash code of
   * the set returned by {@link #edges()}.
   *
   * <p>A reference implementation of this is provided by {@link AbstractGraph#hashCode()}.
   */
  @Override
  int hashCode();
}
