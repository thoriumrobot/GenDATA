// Source-based slice around line 247
// Method: <com.google.common.collect.testing.SafeTreeSet: int hashCode()>

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
    return delegate.toString();
  }

  private static final long serialVersionUID = 0L;
}
