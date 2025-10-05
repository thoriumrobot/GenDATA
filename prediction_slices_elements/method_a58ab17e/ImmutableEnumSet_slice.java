// Source-based slice around line 73
// Method: <com.google.common.collect.ImmutableEnumSet: UnmodifiableIterator iterator()>

    this.delegate = delegate;
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    return Iterators.unmodifiableIterator(delegate.iterator());
  }

  @Override
  public Spliterator<E> spliterator() {
    return delegate.spliterator();
  }

  @Override
  public void forEach(Consumer<? super E> action) {
