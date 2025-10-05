// Source-based slice around line 125
// Method: <com.google.common.collect.testing.SafeTreeSet: E first()>

    return delegate.descendingIterator();
  }

  @Override
  public NavigableSet<E> descendingSet() {
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
