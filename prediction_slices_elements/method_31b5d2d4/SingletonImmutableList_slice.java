// Source-based slice around line 56
// Method: <com.google.common.collect.SingletonImmutableList: Spliterator spliterator()>

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
    return 1;
  }

  @Override
  public ImmutableList<E> subList(int fromIndex, int toIndex) {
