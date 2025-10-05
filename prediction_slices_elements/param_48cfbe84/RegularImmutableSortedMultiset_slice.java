// Source-based slice around line 112
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: ImmutableSortedMultiset tailMultiset(E,BoundType)>

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
    checkPositionIndexes(from, to, length);
    if (from == to) {
      return emptyMultiset(comparator());
    } else if (from == 0 && to == length) {
      return this;
