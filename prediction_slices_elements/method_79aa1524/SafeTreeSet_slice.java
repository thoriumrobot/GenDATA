// Source-based slice around line 91
// Method: <com.google.common.collect.testing.SafeTreeSet: void clear()>

    return delegate.addAll(collection);
  }

  @Override
  public @Nullable E ceiling(E e) {
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
