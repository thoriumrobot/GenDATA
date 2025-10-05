// Source-based slice around line 219
// Method: <com.google.common.collect.Range: Range atMost(C)>

  public static <C extends Comparable<?>> Range<C> lessThan(C endpoint) {
    return create(Cut.belowAll(), Cut.belowValue(endpoint));
  }

  /**
   * Returns a range that contains all values less than or equal to {@code endpoint}.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> atMost(C endpoint) {
    return create(Cut.belowAll(), Cut.aboveValue(endpoint));
  }

  /**
   * Returns a range with no lower bound up to the given endpoint, which may be either inclusive
   * (closed) or exclusive (open).
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> upTo(C endpoint, BoundType boundType) {
