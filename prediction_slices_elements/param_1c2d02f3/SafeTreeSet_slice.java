// Source-based slice around line 86
// Method: <com.google.common.collect.testing.SafeTreeSet: E ceiling(E)>

  @Override
  public boolean addAll(Collection<? extends E> collection) {
    for (E e : collection) {
      checkValid(e);
    }
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
