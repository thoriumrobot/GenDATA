// Source-based slice around line 138
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet copyOf(Iterable)>


  /**
   * Returns an {@code ImmutableRangeSet} containing each of the specified disjoint ranges.
   * Overlapping ranges and empty ranges are forbidden, though adjacent ranges are permitted and
   * will be merged.
   *
   * @throws IllegalArgumentException if any ranges overlap or are empty
   * @since 21.0
   */
  public static <C extends Comparable<?>> ImmutableRangeSet<C> copyOf(Iterable<Range<C>> ranges) {
    return new ImmutableRangeSet.Builder<C>().addAll(ranges).build();
  }

  /**
   * Returns an {@code ImmutableRangeSet} representing the union of the specified ranges.
   *
   * <p>This is the smallest {@code RangeSet} which encloses each of the specified ranges. Duplicate
   * or connected ranges are permitted, and will be coalesced in the result.
   *
   * @since 21.0
