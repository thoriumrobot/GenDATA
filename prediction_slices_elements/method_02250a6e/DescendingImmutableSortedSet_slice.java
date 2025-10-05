// Source-based slice around line 76
// Method: <com.google.common.collect.DescendingImmutableSortedSet: UnmodifiableIterator descendingIterator()>


  @Override
  @GwtIncompatible("NavigableSet")
  public ImmutableSortedSet<E> descendingSet() {
    return forward;
  }

  @Override
  @GwtIncompatible("NavigableSet")
  public UnmodifiableIterator<E> descendingIterator() {
    return forward.iterator();
  }

  @Override
  @GwtIncompatible("NavigableSet")
  ImmutableSortedSet<E> createDescendingSet() {
    throw new AssertionError("should never be called");
  }

  @Override
