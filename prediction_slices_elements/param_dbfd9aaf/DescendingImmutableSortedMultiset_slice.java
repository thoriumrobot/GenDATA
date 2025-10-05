// Source-based slice around line 71
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: ImmutableSortedMultiset headMultiset(E,BoundType)>

    return forward.entrySet().asList().reverse().get(index);
  }

  @Override
  public ImmutableSortedMultiset<E> descendingMultiset() {
    return forward;
  }

  @Override
  public ImmutableSortedMultiset<E> headMultiset(E upperBound, BoundType boundType) {
    return forward.tailMultiset(upperBound, boundType).descendingMultiset();
  }

  @Override
  public ImmutableSortedMultiset<E> tailMultiset(E lowerBound, BoundType boundType) {
    return forward.headMultiset(lowerBound, boundType).descendingMultiset();
  }

  @Override
  boolean isPartialView() {
