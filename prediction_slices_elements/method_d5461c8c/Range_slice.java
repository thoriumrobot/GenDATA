// Source-based slice around line 385
// Method: <com.google.common.collect.Range: BoundType upperBoundType()>

  }

  /**
   * Returns the type of this range's upper bound: {@link BoundType#CLOSED} if the range includes
   * its upper endpoint, {@link BoundType#OPEN} if it does not.
   *
   * @throws IllegalStateException if this range is unbounded above (that is, {@link
   *     #hasUpperBound()} returns {@code false})
   */
  public BoundType upperBoundType() {
    return upperBound.typeAsUpperBound();
  }

  /**
   * Returns {@code true} if this range is of the form {@code [v..v)} or {@code (v..v]}. (This does
   * not encompass ranges of the form {@code (v..v)}, because such ranges are <i>invalid</i> and
   * can't be constructed at all.)
   *
   * <p>Note that certain discrete ranges such as the integer range {@code (3..4)} are <b>not</b>
   * considered empty, even though they contain no actual values. In these cases, it may be helpful
