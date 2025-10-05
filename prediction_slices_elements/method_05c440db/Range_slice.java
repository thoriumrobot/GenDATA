// Source-based slice around line 663
// Method: <com.google.common.collect.Range: boolean equals(Object)>


  /**
   * Returns {@code true} if {@code object} is a range having the same endpoints and bound types as
   * this range. Note that discrete ranges such as {@code (1..4)} and {@code [2..3]} are <b>not</b>
   * equal to one another, despite the fact that they each contain precisely the same set of values.
   * Similarly, empty ranges are not equal unless they have exactly the same representation, so
   * {@code [3..3)}, {@code (3..3]}, {@code (4..4]} are all unequal.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Range) {
      Range<?> other = (Range<?>) object;
      return lowerBound.equals(other.lowerBound) && upperBound.equals(other.upperBound);
    }
    return false;
  }

  /** Returns a hash code for this range. */
  @Override
  public int hashCode() {
