// Source-based slice around line 241
// Method: <com.google.common.collect.ContiguousSet: Range range(BoundType,BoundType)>

   * {@linkplain Range#contains(Comparable) contained} within the range.
   *
   * <p>Note that this method will return ranges with unbounded endpoints if {@link BoundType#OPEN}
   * is requested for a domain minimum or maximum. For example, if {@code set} was created from the
   * range {@code [1..Integer.MAX_VALUE]} then {@code set.range(CLOSED, OPEN)} must return {@code
   * [1..∞)}.
   *
   * @throws NoSuchElementException if this set is empty
   */
  public abstract Range<C> range(BoundType lowerBoundType, BoundType upperBoundType);

  @Override
  @GwtIncompatible // NavigableSet
  ImmutableSortedSet<C> createDescendingSet() {
    return new DescendingImmutableSortedSet<>(this);
  }

  /** Returns a shorthand representation of the contents such as {@code "[1..100]"}. */
  @Override
  public String toString() {
