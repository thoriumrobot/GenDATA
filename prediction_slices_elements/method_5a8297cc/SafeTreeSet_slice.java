// Source-based slice around line 223
// Method: <com.google.common.collect.testing.SafeTreeSet: Object[] toArray()>

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
    return delegate.toArray(a);
  }

  @CanIgnoreReturnValue
  private <T> T checkValid(T t) {
