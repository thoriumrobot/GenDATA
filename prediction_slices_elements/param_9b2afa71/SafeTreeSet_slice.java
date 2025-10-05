// Source-based slice around line 228
// Method: <com.google.common.collect.testing.SafeTreeSet: T[] toArray(T[])>

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
    // a ClassCastException is what's supposed to happen!
    @SuppressWarnings("unchecked")
    E e = (E) t;
    int unused = comparator().compare(e, e);
    return t;
