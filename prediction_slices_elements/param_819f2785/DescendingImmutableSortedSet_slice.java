// Source-based slice around line 87
// Method: <com.google.common.collect.DescendingImmutableSortedSet: E lower(E)>

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
    return forward.ceiling(element);
  }

  @Override
  public @Nullable E ceiling(E element) {
