// Source-based slice around line 193
// Method: <com.google.common.collect.Range: Range range(C,BoundType,C,BoundType)>


  /**
   * Returns a range that contains any value from {@code lower} to {@code upper}, where each
   * endpoint may be either inclusive (closed) or exclusive (open).
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> range(
      C lower, BoundType lowerType, C upper, BoundType upperType) {
    checkNotNull(lowerType);
    checkNotNull(upperType);

    Cut<C> lowerBound =
        (lowerType == BoundType.OPEN) ? Cut.aboveValue(lower) : Cut.belowValue(lower);
    Cut<C> upperBound =
        (upperType == BoundType.OPEN) ? Cut.belowValue(upper) : Cut.aboveValue(upper);
    return create(lowerBound, upperBound);
  }
