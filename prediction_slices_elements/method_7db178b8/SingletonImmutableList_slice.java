// Source-based slice around line 61
// Method: <com.google.common.collect.SingletonImmutableList: int size()>

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
    Preconditions.checkPositionIndexes(fromIndex, toIndex, 1);
    return (fromIndex == toIndex) ? ImmutableList.of() : this;
  }

  @Override
