// Source-based slice around line 81
// Method: <com.google.common.collect.RangeSet: boolean encloses(Range)>

   *
   * @since 20.0
   */
  boolean intersects(Range<C> otherRange);

  /**
   * Returns {@code true} if there exists a member range in this range set which {@linkplain
   * Range#encloses encloses} the specified range.
   */
  boolean encloses(Range<C> otherRange);

  /**
   * Returns {@code true} if for each member range in {@code other} there exists a member range in
   * this range set which {@linkplain Range#encloses encloses} it. It follows that {@code
   * this.contains(value)} whenever {@code other.contains(value)}. Returns {@code true} if {@code
   * other} is empty.
   *
   * <p>This is equivalent to checking if this range set {@link #encloses} each of the ranges in
   * {@code other}.
   */
