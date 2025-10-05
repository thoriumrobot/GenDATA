// Source-based slice around line 364
// Method: <com.google.common.collect.Range: boolean hasUpperBound()>

   *
   * @throws IllegalStateException if this range is unbounded below (that is, {@link
   *     #hasLowerBound()} returns {@code false})
   */
  public BoundType lowerBoundType() {
    return lowerBound.typeAsLowerBound();
  }

  /** Returns {@code true} if this range has an upper endpoint. */
  public boolean hasUpperBound() {
    return upperBound != Cut.aboveAll();
  }

  /**
   * Returns the upper endpoint of this range.
   *
   * @throws IllegalStateException if this range is unbounded above (that is, {@link
   *     #hasUpperBound()} returns {@code false})
   */
  public C upperEndpoint() {
