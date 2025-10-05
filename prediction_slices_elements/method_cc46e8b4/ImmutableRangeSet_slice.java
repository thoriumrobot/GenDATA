// Source-based slice around line 423
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet union(RangeSet)>


  /**
   * Returns a new range set consisting of the union of this range set and {@code other}.
   *
   * <p>This is essentially the same as {@code TreeRangeSet.create(this).addAll(other)} except it
   * returns an {@code ImmutableRangeSet}.
   *
   * @since 21.0
   */
  public ImmutableRangeSet<C> union(RangeSet<C> other) {
    return unionOf(Iterables.concat(asRanges(), other.asRanges()));
  }

  /**
   * Returns a new range set consisting of the intersection of this range set and {@code other}.
   *
   * <p>This is essentially the same as {@code
   * TreeRangeSet.create(this).removeAll(other.complement())} except it returns an {@code
   * ImmutableRangeSet}.
   *
