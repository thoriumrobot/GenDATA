// Source-based slice around line 82
// Method: <com.google.common.collect.DescendingImmutableSortedSet: ImmutableSortedSet createDescendingSet()>


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
  public @Nullable E lower(E element) {
    return forward.higher(element);
  }

  @Override
  public @Nullable E floor(E element) {
