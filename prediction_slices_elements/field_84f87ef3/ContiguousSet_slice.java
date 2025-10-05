// Source-based slice around line 145
// Method: com.google.common.collect.ContiguousSet.domain

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

  @Override
  public ContiguousSet<C> headSet(C toElement) {
    return headSetImpl(checkNotNull(toElement), false);
  }
