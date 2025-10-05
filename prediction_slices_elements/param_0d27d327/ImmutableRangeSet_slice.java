// Source-based slice around line 562
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableSortedSet asSet(DiscreteDomain)>

   * such a set can be performed efficiently, but others (such as {@link Set#hashCode} or {@link
   * Collections#frequency}) can cause major performance problems.
   *
   * <p>The returned set's {@link Object#toString} method returns a shorthand form of the set's
   * contents, such as {@code "[1..100]}"}.
   *
   * @throws IllegalArgumentException if neither this range nor the domain has a lower bound, or if
   *     neither has an upper bound
   */
  public ImmutableSortedSet<C> asSet(DiscreteDomain<C> domain) {
    checkNotNull(domain);
    if (isEmpty()) {
      return ImmutableSortedSet.of();
    }
    Range<C> span = span().canonical(domain);
    if (!span.hasLowerBound()) {
      // according to the spec of canonical, neither this ImmutableRangeSet nor
      // the range have a lower bound
      throw new IllegalArgumentException(
          "Neither the DiscreteDomain nor this range set are bounded below");
