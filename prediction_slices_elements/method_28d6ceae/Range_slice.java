// Source-based slice around line 291
// Method: <com.google.common.collect.Range: Range singleton(C)>

    return (Range) ALL;
  }

  /**
   * Returns a range that {@linkplain Range#contains(Comparable) contains} only the given value. The
   * returned range is {@linkplain BoundType#CLOSED closed} on both ends.
   *
   * @since 14.0
   */
  public static <C extends Comparable<?>> Range<C> singleton(C value) {
    return closed(value, value);
  }

  /**
   * Returns the minimal range that {@linkplain Range#contains(Comparable) contains} all of the
   * given values. The returned range is {@linkplain BoundType#CLOSED closed} on both ends.
   *
   * @throws ClassCastException if the values are not mutually comparable
   * @throws NoSuchElementException if {@code values} is empty
   * @throws NullPointerException if any of {@code values} is null
