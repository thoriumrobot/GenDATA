// Source-based slice around line 117
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet closed(long,long)>


  /**
   * Returns a nonempty contiguous set containing all {@code long} values from {@code lower}
   * (inclusive) to {@code upper} (inclusive). (These are the same values contained in {@code
   * Range.closed(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
   */
  public static ContiguousSet<Long> closed(long lower, long upper) {
    return create(Range.closed(lower, upper), DiscreteDomain.longs());
  }

  /**
   * Returns a contiguous set containing all {@code int} values from {@code lower} (inclusive) to
   * {@code upper} (exclusive). If the endpoints are equal, an empty set is returned. (These are the
   * same values contained in {@code Range.closedOpen(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
