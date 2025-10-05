// Source-based slice around line 190
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean retainAll(Collection)>

    return delegate.remove(checkValid(object));
  }

  @Override
  public boolean removeAll(Collection<?> c) {
    return delegate.removeAll(c);
  }

  @Override
  public boolean retainAll(Collection<?> c) {
    return delegate.retainAll(c);
  }

  @Override
  public int size() {
    return delegate.size();
  }

  @Override
  public NavigableSet<E> subSet(
