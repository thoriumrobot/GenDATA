// Source-based slice around line 51
// Method: <com.google.common.collect.SingletonImmutableList: UnmodifiableIterator iterator()>

  }

  @Override
  public E get(int index) {
    Preconditions.checkElementIndex(index, 1);
    return element;
  }

  @Override
  public UnmodifiableIterator<E> iterator() {
    return singletonIterator(element);
  }

  @Override
  public Spliterator<E> spliterator() {
    return singleton(element).spliterator();
  }

  @Override
  public int size() {
