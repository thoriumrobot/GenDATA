// Source-based slice around line 181
// Method: <com.google.common.collect.Range: Range openClosed(C,C)>


  /**
   * Returns a range that contains all values strictly greater than {@code lower} and less than or
   * equal to {@code upper}.
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> openClosed(C lower, C upper) {
    return create(Cut.aboveValue(lower), Cut.aboveValue(upper));
  }

  /**
   * Returns a range that contains any value from {@code lower} to {@code upper}, where each
   * endpoint may be either inclusive (closed) or exclusive (open).
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @throws ClassCastException if {@code lower} and {@code upper} are not mutually comparable
   * @since 14.0
