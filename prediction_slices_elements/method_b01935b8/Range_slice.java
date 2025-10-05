// Source-based slice around line 515
// Method: <com.google.common.collect.Range: boolean isConnected(Range)>

   *
   * <p>The connectedness relation is both reflexive and symmetric, but does not form an {@linkplain
   * Equivalence equivalence relation} as it is not transitive.
   *
   * <p>Note that certain discrete ranges are not considered connected, even though there are no
   * elements "between them." For example, {@code [3, 5]} is not considered connected to {@code [6,
   * 10]}. In these cases, it may be desirable for both input ranges to be preprocessed with {@link
   * #canonical(DiscreteDomain)} before testing for connectedness.
   */
  public boolean isConnected(Range<C> other) {
    return lowerBound.compareTo(other.upperBound) <= 0
        && other.lowerBound.compareTo(upperBound) <= 0;
  }

  /**
   * Returns the maximal range {@linkplain #encloses enclosed} by both this range and {@code
   * connectedRange}, if such a range exists.
   *
   * <p>For example, the intersection of {@code [1..5]} and {@code (3..7)} is {@code (3..5]}. The
   * resulting range may be empty; for example, {@code [1..5)} intersected with {@code [5..7)}
