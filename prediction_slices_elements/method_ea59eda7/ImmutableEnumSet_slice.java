// Source-based slice around line 88
// Method: <com.google.common.collect.ImmutableEnumSet: int size()>

    return delegate.spliterator();
  }

  @Override
  public void forEach(Consumer<? super E> action) {
    delegate.forEach(action);
  }

  @Override
  public int size() {
    return delegate.size();
  }

  @Override
  public boolean contains(@Nullable Object object) {
    return delegate.contains(object);
  }

  @Override
  public boolean containsAll(Collection<?> collection) {
