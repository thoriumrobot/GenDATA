// Source-based slice around line 63
// Method: <com.google.common.collect.ContiguousSet: ContiguousSet create(Range,DiscreteDomain)>

  /**
   * Returns a {@code ContiguousSet} containing the same values in the given domain {@linkplain
   * Range#contains contained} by the range.
   *
   * @throws IllegalArgumentException if neither range nor the domain has a lower bound, or if
   *     neither has an upper bound
   * @since 13.0
   */
  public static <C extends Comparable> ContiguousSet<C> create(
      Range<C> range, DiscreteDomain<C> domain) {
    checkNotNull(range);
    checkNotNull(domain);
    Range<C> effectiveRange = range;
    try {
      if (!range.hasLowerBound()) {
        effectiveRange = effectiveRange.intersection(Range.atLeast(domain.minValue()));
      }
      if (!range.hasUpperBound()) {
        effectiveRange = effectiveRange.intersection(Range.atMost(domain.maxValue()));
      }
