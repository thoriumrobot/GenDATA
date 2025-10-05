// Source-based slice around line 450
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet difference(RangeSet)>


  /**
   * Returns a new range set consisting of the difference of this range set and {@code other}.
   *
   * <p>This is essentially the same as {@code TreeRangeSet.create(this).removeAll(other)} except it
   * returns an {@code ImmutableRangeSet}.
   *
   * @since 21.0
   */
  public ImmutableRangeSet<C> difference(RangeSet<C> other) {
    RangeSet<C> copy = TreeRangeSet.create(this);
    copy.removeAll(other);
    return copyOf(copy);
  }

  /**
   * Returns a list containing the nonempty intersections of {@code range} with the ranges in this
   * range set.
   */
  private ImmutableList<Range<C>> intersectRanges(Range<C> range) {
