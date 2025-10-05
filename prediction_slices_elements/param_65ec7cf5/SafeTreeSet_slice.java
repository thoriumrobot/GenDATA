// Source-based slice around line 145
// Method: <com.google.common.collect.testing.SafeTreeSet: E higher(E)>

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
    return delegate.isEmpty();
  }

  @Override
  public Iterator<E> iterator() {
