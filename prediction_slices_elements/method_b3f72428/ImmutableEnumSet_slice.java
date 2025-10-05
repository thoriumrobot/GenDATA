// Source-based slice around line 78
// Method: <com.google.common.collect.ImmutableEnumSet: Spliterator spliterator()>

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
    delegate.forEach(action);
  }

  @Override
  public int size() {
