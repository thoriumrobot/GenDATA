// Source-based slice around line 436
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet intersection(RangeSet)>

  /**
   * Returns a new range set consisting of the intersection of this range set and {@code other}.
   *
   * <p>This is essentially the same as {@code
   * TreeRangeSet.create(this).removeAll(other.complement())} except it returns an {@code
   * ImmutableRangeSet}.
   *
   * @since 21.0
   */
  public ImmutableRangeSet<C> intersection(RangeSet<C> other) {
    RangeSet<C> copy = TreeRangeSet.create(this);
    copy.removeAll(other.complement());
    return copyOf(copy);
  }

  /**
   * Returns a new range set consisting of the difference of this range set and {@code other}.
   *
   * <p>This is essentially the same as {@code TreeRangeSet.create(this).removeAll(other)} except it
   * returns an {@code ImmutableRangeSet}.
