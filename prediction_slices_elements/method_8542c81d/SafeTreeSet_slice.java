// Source-based slice around line 115
// Method: <com.google.common.collect.testing.SafeTreeSet: Iterator descendingIterator()>

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
    return new SafeTreeSet<>(delegate.descendingSet());
  }

  @Override
  public E first() {
