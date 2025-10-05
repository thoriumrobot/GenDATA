// Source-based slice around line 242
// Method: <com.google.common.collect.testing.SafeTreeSet: boolean equals(Object)>

  private <T> T checkValid(T t) {
    // a ClassCastException is what's supposed to happen!
    @SuppressWarnings("unchecked")
    E e = (E) t;
    int unused = comparator().compare(e, e);
    return t;
  }

  @Override
  public boolean equals(@Nullable Object obj) {
    return delegate.equals(obj);
  }

  @Override
  public int hashCode() {
    return delegate.hashCode();
  }

  @Override
  public String toString() {
