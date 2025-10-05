// Source-based slice around line 114
// Method: <com.google.common.collect.RangeSet: boolean isEmpty()>

    for (Range<C> range : other) {
      if (!encloses(range)) {
        return false;
      }
    }
    return true;
  }

  /** Returns {@code true} if this range set contains no ranges. */
  boolean isEmpty();

  /**
   * Returns the minimal range which {@linkplain Range#encloses(Range) encloses} all ranges in this
   * range set.
   *
   * @throws NoSuchElementException if this range set is {@linkplain #isEmpty() empty}
   */
  Range<C> span();

  // Views
