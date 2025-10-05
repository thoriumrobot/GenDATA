// Source-based slice around line 245
// Method: <com.google.common.collect.ContiguousSet: ImmutableSortedSet createDescendingSet()>

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
    return range().toString();
  }

  /**
