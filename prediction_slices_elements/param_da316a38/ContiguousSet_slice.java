// Source-based slice around line 141
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet closedOpen(long,long)>


  /**
   * Returns a contiguous set containing all {@code long} values from {@code lower} (inclusive) to
   * {@code upper} (exclusive). If the endpoints are equal, an empty set is returned. (These are the
   * same values contained in {@code Range.closedOpen(lower, upper)}.)
   *
   * @throws IllegalArgumentException if {@code lower} is greater than {@code upper}
   * @since 23.0
   */
  public static ContiguousSet<Long> closedOpen(long lower, long upper) {
    return create(Range.closedOpen(lower, upper), DiscreteDomain.longs());
  }

  final DiscreteDomain<C> domain;

  ContiguousSet(DiscreteDomain<C> domain) {
    super(Ordering.natural());
    this.domain = domain;
  }

