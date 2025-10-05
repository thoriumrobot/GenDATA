// Source-based slice around line 338
// Method: <com.google.common.collect.Range: boolean hasLowerBound()>

    this.upperBound = checkNotNull(upperBound);
    if (lowerBound.compareTo(upperBound) > 0
        || lowerBound == Cut.<C>aboveAll()
        || upperBound == Cut.<C>belowAll()) {
      throw new IllegalArgumentException("Invalid range: " + toString(lowerBound, upperBound));
    }
  }

  /** Returns {@code true} if this range has a lower endpoint. */
  public boolean hasLowerBound() {
    return lowerBound != Cut.belowAll();
  }

  /**
   * Returns the lower endpoint of this range.
   *
   * @throws IllegalStateException if this range is unbounded below (that is, {@link
   *     #hasLowerBound()} returns {@code false})
   */
  public C lowerEndpoint() {
