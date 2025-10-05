// Source-based slice around line 253
// Method: <com.google.common.collect.Range: Range atLeast(C)>

  public static <C extends Comparable<?>> Range<C> greaterThan(C endpoint) {
    return create(Cut.aboveValue(endpoint), Cut.aboveAll());
  }

  /**
   * Returns a range that contains all values greater than or equal to {@code endpoint}.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> atLeast(C endpoint) {
    return create(Cut.belowValue(endpoint), Cut.aboveAll());
  }

  /**
   * Returns a range from the given endpoint, which may be either inclusive (closed) or exclusive
   * (open), with no upper bound.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> downTo(C endpoint, BoundType boundType) {
