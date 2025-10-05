// Source-based slice around line 120
// Method: <com.google.common.collect.testing.SafeTreeSet: NavigableSet descendingSet()>

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
    return delegate.first();
  }

  @Override
  public @Nullable E floor(E e) {
