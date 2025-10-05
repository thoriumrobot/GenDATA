// Source-based slice around line 165
// Method: com.google.common.collect.ImmutableRangeSet.complement

  }

  private ImmutableRangeSet(
      ImmutableList<Range<C>> ranges, @Nullable ImmutableRangeSet<C> complement) {
    this.ranges = ranges;
    this.complement = complement;
  }

  private final transient ImmutableList<Range<C>> ranges;
  private final transient @Nullable ImmutableRangeSet<C> complement;

  @Override
  public boolean intersects(Range<C> otherRange) {
    int ceilingIndex =
        SortedLists.binarySearch(
            ranges,
            Range::lowerBound,
            otherRange.lowerBound,
            Ordering.natural(),
            ANY_PRESENT,
