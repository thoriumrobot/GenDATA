// Source-based slice around line 61
// Method: <com.google.common.collect.EmptyContiguousSet: Range range()>

    return 0;
  }

  @Override
  public ContiguousSet<C> intersection(ContiguousSet<C> other) {
    return this;
  }

  @Override
  public Range<C> range() {
    throw new NoSuchElementException();
  }

  @Override
  public Range<C> range(BoundType lowerBoundType, BoundType upperBoundType) {
    throw new NoSuchElementException();
  }

  @Override
  ContiguousSet<C> headSetImpl(C toElement, boolean inclusive) {
