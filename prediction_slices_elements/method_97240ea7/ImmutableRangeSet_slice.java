// Source-based slice around line 76
// Method: <com.google.common.collect.ImmutableRangeSet: Collector toImmutableRangeSet()>

      new ImmutableRangeSet<>(ImmutableList.of(Range.all()));

  /**
   * Returns a {@code Collector} that accumulates the input elements into a new {@code
   * ImmutableRangeSet}. As in {@link Builder}, overlapping ranges are not permitted and adjacent
   * ranges will be merged.
   *
   * @since 23.1
   */
  public static <E extends Comparable<? super E>>
      Collector<Range<E>, ?, ImmutableRangeSet<E>> toImmutableRangeSet() {
    return CollectCollectors.toImmutableRangeSet();
  }

  /**
   * Returns an empty immutable range set.
   *
   * <p><b>Performance note:</b> the instance returned is a singleton.
   */
  @SuppressWarnings("unchecked")
