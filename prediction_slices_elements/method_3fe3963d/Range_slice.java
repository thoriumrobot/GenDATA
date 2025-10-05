// Source-based slice around line 348
// Method: <com.google.common.collect.Range: C lowerEndpoint()>

    return lowerBound != Cut.belowAll();
  }

  /**
   * Returns the lower endpoint of this range.
   *
   * @throws IllegalStateException if this range is unbounded below (that is, {@link
   *     #hasLowerBound()} returns {@code false})
   */
  public C lowerEndpoint() {
    return lowerBound.endpoint();
  }

  /**
   * Returns the type of this range's lower bound: {@link BoundType#CLOSED} if the range includes
   * its lower endpoint, {@link BoundType#OPEN} if it does not.
   *
   * @throws IllegalStateException if this range is unbounded below (that is, {@link
   *     #hasLowerBound()} returns {@code false})
   */
