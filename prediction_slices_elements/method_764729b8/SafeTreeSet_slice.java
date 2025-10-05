// Source-based slice around line 105
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean contains(Object)>

  public Comparator<? super E> comparator() {
    Comparator<? super E> comparator = delegate.comparator();
    if (comparator == null) {
      comparator = (Comparator<? super E>) NATURAL_ORDER;
    }
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
