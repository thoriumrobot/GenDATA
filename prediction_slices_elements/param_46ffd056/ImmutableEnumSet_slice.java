// Source-based slice around line 83
// Method: <com.google.common.collect.ImmutableEnumSet: void forEach(Consumer)>

    return Iterators.unmodifiableIterator(delegate.iterator());
  }

  @Override
  public Spliterator<E> spliterator() {
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
