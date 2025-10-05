// Source-based slice around line 150
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet unionOf(Iterable)>


  /**
   * Returns an {@code ImmutableRangeSet} representing the union of the specified ranges.
   *
   * <p>This is the smallest {@code RangeSet} which encloses each of the specified ranges. Duplicate
   * or connected ranges are permitted, and will be coalesced in the result.
   *
   * @since 21.0
   */
  public static <C extends Comparable<?>> ImmutableRangeSet<C> unionOf(Iterable<Range<C>> ranges) {
    return copyOf(TreeRangeSet.create(ranges));
  }

  ImmutableRangeSet(ImmutableList<Range<C>> ranges) {
    this(ranges, /* complement= */ null);
  }

  private ImmutableRangeSet(
      ImmutableList<Range<C>> ranges, @Nullable ImmutableRangeSet<C> complement) {
    this.ranges = ranges;
