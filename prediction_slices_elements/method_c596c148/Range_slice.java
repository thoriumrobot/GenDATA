// Source-based slice around line 374
// Method: <com.google.common.collect.Range: C upperEndpoint()>

    return upperBound != Cut.aboveAll();
  }

  /**
   * Returns the upper endpoint of this range.
   *
   * @throws IllegalStateException if this range is unbounded above (that is, {@link
   *     #hasUpperBound()} returns {@code false})
   */
  public C upperEndpoint() {
    return upperBound.endpoint();
  }

  /**
   * Returns the type of this range's upper bound: {@link BoundType#CLOSED} if the range includes
   * its upper endpoint, {@link BoundType#OPEN} if it does not.
   *
   * @throws IllegalStateException if this range is unbounded above (that is, {@link
   *     #hasUpperBound()} returns {@code false})
   */
