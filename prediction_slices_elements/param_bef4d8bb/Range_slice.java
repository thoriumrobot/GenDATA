// Source-based slice around line 210
// Method: <com.google.common.collect.Range: Range lessThan(C)>

        (upperType == BoundType.OPEN) ? Cut.belowValue(upper) : Cut.aboveValue(upper);
    return create(lowerBound, upperBound);
  }

  /**
   * Returns a range that contains all values strictly less than {@code endpoint}.
   *
   * @since 14.0
   */
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
