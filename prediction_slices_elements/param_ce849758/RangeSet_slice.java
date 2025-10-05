// Source-based slice around line 92
// Method: <com.google.common.collect.RangeSet: boolean enclosesAll(RangeSet)>

  /**
   * Returns {@code true} if for each member range in {@code other} there exists a member range in
   * this range set which {@linkplain Range#encloses encloses} it. It follows that {@code
   * this.contains(value)} whenever {@code other.contains(value)}. Returns {@code true} if {@code
   * other} is empty.
   *
   * <p>This is equivalent to checking if this range set {@link #encloses} each of the ranges in
   * {@code other}.
   */
  boolean enclosesAll(RangeSet<C> other);

  /**
   * Returns {@code true} if for each range in {@code other} there exists a member range in this
   * range set which {@linkplain Range#encloses encloses} it. Returns {@code true} if {@code other}
   * is empty.
   *
   * <p>This is equivalent to checking if this range set {@link #encloses} each range in {@code
   * other}.
   *
   * @since 21.0
