// Source-based slice around line 180
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean remove(Object)>

    return delegate.pollFirst();
  }

  @Override
  public @Nullable E pollLast() {
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
