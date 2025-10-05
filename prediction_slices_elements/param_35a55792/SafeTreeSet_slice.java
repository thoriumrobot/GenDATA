// Source-based slice around line 140
// Method: <com.google.common.collect.testing.SafeTreeSet: NavigableSet headSet(E,boolean)>

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
    return delegate.higher(checkValid(e));
  }

  @Override
  public boolean isEmpty() {
