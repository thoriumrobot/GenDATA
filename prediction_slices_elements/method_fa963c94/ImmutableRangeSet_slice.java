// Source-based slice around line 87
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet of()>

    return CollectCollectors.toImmutableRangeSet();
  }

  /**
   * Returns an empty immutable range set.
   *
   * <p><b>Performance note:</b> the instance returned is a singleton.
   */
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
