// Source-based slice around line 117
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: ImmutableSortedMultiset getSubMultiset(int,int)>

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
    } else {
      RegularImmutableSortedSet<E> subElementSet = elementSet.getSubSet(from, to);
      return new RegularImmutableSortedMultiset<>(
          subElementSet, cumulativeCounts, offset + from, to - from);
    }
