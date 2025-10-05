// Source-based slice around line 129
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet closedOpen(int,int)>


  /**
   * Returns a contiguous set containing all {@code int} values from {@code lower} (inclusive) to
   * {@code upper} (exclusive). If the endpoints are equal, an empty set is returned. (These are the
   * same values contained in {@code Range.closedOpen(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
   */
  public static ContiguousSet<Integer> closedOpen(int lower, int upper) {
    return create(Range.closedOpen(lower, upper), DiscreteDomain.integers());
  }

  /**
   * Returns a contiguous set containing all {@code long} values from {@code lower} (inclusive) to
   * {@code upper} (exclusive). If the endpoints are equal, an empty set is returned. (These are the
   * same values contained in {@code Range.closedOpen(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
