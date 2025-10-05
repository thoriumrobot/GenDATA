// Source-based slice around line 135
// Method: <com.google.common.collect.testing.SafeTreeSet: SortedSet headSet(E)>

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
    return new SafeTreeSet<>(delegate.headSet(checkValid(toElement), inclusive));
  }

  @Override
  public @Nullable E higher(E e) {
