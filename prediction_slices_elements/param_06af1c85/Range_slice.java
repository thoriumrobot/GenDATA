// Source-based slice around line 263
// Method: <com.google.common.collect.Range: Range downTo(C,BoundType)>

    return create(Cut.belowValue(endpoint), Cut.aboveAll());
  }

  /**
   * Returns a range from the given endpoint, which may be either inclusive (closed) or exclusive
   * (open), with no upper bound.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> downTo(C endpoint, BoundType boundType) {
    switch (boundType) {
      case OPEN:
        return greaterThan(endpoint);
      case CLOSED:
        return atLeast(endpoint);
    }
    throw new AssertionError();
  }

  private static final Range<Comparable> ALL = new Range<>(Cut.belowAll(), Cut.aboveAll());
