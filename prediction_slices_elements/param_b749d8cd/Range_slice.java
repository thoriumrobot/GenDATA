// Source-based slice around line 145
// Method: <com.google.common.collect.Range: Range open(C,C)>

  /**
   * Returns a range that contains all values strictly greater than {@code lower} and strictly less
   * than {@code upper}.
   *
   * @throws IllegalArgumentException if {@code lower} is greater than <i>or equal to</i> {@code
   *     upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> open(C lower, C upper) {
    return create(Cut.aboveValue(lower), Cut.belowValue(upper));
  }

  /**
   * Returns a range that contains all values greater than or equal to {@code lower} and less than
   * or equal to {@code upper}.
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
   * @since 14.0
