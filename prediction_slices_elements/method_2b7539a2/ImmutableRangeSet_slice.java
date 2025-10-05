// Source-based slice around line 95
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet of(Range)>

  @SuppressWarnings("unchecked")
  public static <C extends Comparable> ImmutableRangeSet<C> of() {
    return (ImmutableRangeSet<C>) EMPTY;
  }

  /**
   * Returns an immutable range set containing the specified single range. If {@link Range#isEmpty()
   * range.isEmpty()}, this is equivalent to {@link ImmutableRangeSet#of()}.
   */
  public static <C extends Comparable> ImmutableRangeSet<C> of(Range<C> range) {
    checkNotNull(range);
    if (range.isEmpty()) {
      return of();
    } else if (range.equals(Range.all())) {
      return all();
    } else {
      return new ImmutableRangeSet<>(ImmutableList.of(range));
    }
  }

