// Source-based slice around line 105
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet closed(int,int)>


  /**
   * Returns a nonempty contiguous set containing all {@code int} values from {@code lower}
   * (inclusive) to {@code upper} (inclusive). (These are the same values contained in {@code
   * Range.closed(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
   */
  public static ContiguousSet<Integer> closed(int lower, int upper) {
    return create(Range.closed(lower, upper), DiscreteDomain.integers());
  }

  /**
   * Returns a nonempty contiguous set containing all {@code long} values from {@code lower}
   * (inclusive) to {@code upper} (inclusive). (These are the same values contained in {@code
   * Range.closed(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
