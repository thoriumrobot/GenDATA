// Source-based slice around line 76
// Method: <com.google.common.collect.DescendingImmutableSortedMultiset: ImmutableSortedMultiset tailMultiset(E,BoundType)>

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
    return forward.isPartialView();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
