// Source-based slice around line 110
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean containsAll(Collection)>

    return comparator;
  }

  @Override
  public boolean contains(Object object) {
    return delegate.contains(checkValid(object));
  }

  @Override
  public boolean containsAll(Collection<?> c) {
    return delegate.containsAll(c);
  }

  @Override
  public Iterator<E> descendingIterator() {
    return delegate.descendingIterator();
  }

  @Override
  public NavigableSet<E> descendingSet() {
