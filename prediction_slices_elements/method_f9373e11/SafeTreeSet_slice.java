// Source-based slice around line 185
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean removeAll(Collection)>

    return delegate.pollLast();
  }

  @Override
  public boolean remove(Object object) {
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
