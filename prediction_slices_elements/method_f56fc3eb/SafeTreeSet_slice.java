// Source-based slice around line 96
// Method: <com.google.common.collect.testing.SafeTreeSet: Comparator comparator()>

    return delegate.ceiling(checkValid(e));
  }

  @Override
  public void clear() {
    delegate.clear();
  }

  @Override
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
