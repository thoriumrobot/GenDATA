// Source-based slice around line 244
// Method: <com.google.common.collect.Range: Range greaterThan(C)>

    }
    throw new AssertionError();
  }

  /**
   * Returns a range that contains all values strictly greater than {@code endpoint}.
   *
   * @since 14.0
   */
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
