// Source-based slice around line 66
// Method: <com.google.common.collect.SingletonImmutableList: ImmutableList subList(int,int)>

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
  public String toString() {
    return '[' + element.toString() + ']';
  }

  @Override
