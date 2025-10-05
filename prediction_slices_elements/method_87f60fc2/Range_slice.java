// Source-based slice around line 359
// Method: <com.google.common.collect.Range: BoundType lowerBoundType()>

  }

  /**
   * Returns the type of this range's lower bound: {@link BoundType#CLOSED} if the range includes
   * its lower endpoint, {@link BoundType#OPEN} if it does not.
   *
   * @throws IllegalStateException if this range is unbounded below (that is, {@link
   *     #hasLowerBound()} returns {@code false})
   */
  public BoundType lowerBoundType() {
    return lowerBound.typeAsLowerBound();
  }

  /** Returns {@code true} if this range has an upper endpoint. */
  public boolean hasUpperBound() {
    return upperBound != Cut.aboveAll();
  }

  /**
   * Returns the upper endpoint of this range.
