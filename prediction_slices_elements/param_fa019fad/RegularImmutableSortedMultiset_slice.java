// Source-based slice around line 107
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: ImmutableSortedMultiset headMultiset(E,BoundType)>

    return Ints.saturatedCast(size);
  }

  @Override
  public ImmutableSortedSet<E> elementSet() {
    return elementSet;
  }

  @Override
  public ImmutableSortedMultiset<E> headMultiset(E upperBound, BoundType boundType) {
    return getSubMultiset(0, elementSet.headIndex(upperBound, checkNotNull(boundType) == CLOSED));
  }

  @Override
  public ImmutableSortedMultiset<E> tailMultiset(E lowerBound, BoundType boundType) {
    return getSubMultiset(
        elementSet.tailIndex(lowerBound, checkNotNull(boundType) == CLOSED), length);
  }

  ImmutableSortedMultiset<E> getSubMultiset(int from, int to) {
