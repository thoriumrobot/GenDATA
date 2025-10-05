// Source-based slice around line 218
// Method: <com.google.common.collect.testing.SafeTreeSet: NavigableSet tailSet(E,boolean)>

    return subSet(fromElement, true, toElement, false);
  }

  @Override
  public SortedSet<E> tailSet(E fromElement) {
    return tailSet(fromElement, true);
  }

  @Override
  public NavigableSet<E> tailSet(E fromElement, boolean inclusive) {
    return new SafeTreeSet<>(delegate.tailSet(checkValid(fromElement), inclusive));
  }

  @Override
  public Object[] toArray() {
    return delegate.toArray();
  }

  @Override
  public <T> T[] toArray(T[] a) {
