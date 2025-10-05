// Source-based slice around line 102
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: ImmutableSortedSet elementSet()>

  }

  @Override
  public int size() {
    long size = cumulativeCounts[offset + length] - cumulativeCounts[offset];
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
