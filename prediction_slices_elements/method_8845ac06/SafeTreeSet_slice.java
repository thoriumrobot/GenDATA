// Source-based slice around line 130
// Method: <com.google.common.collect.testing.SafeTreeSet: E floor(E)>

    return new SafeTreeSet<>(delegate.descendingSet());
  }

  @Override
  public E first() {
    return delegate.first();
  }

  @Override
  public @Nullable E floor(E e) {
    return delegate.floor(checkValid(e));
  }

  @Override
  public SortedSet<E> headSet(E toElement) {
    return headSet(toElement, false);
  }

  @Override
  public NavigableSet<E> headSet(E toElement, boolean inclusive) {
