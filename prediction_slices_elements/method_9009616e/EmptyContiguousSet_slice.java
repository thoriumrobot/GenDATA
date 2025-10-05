// Source-based slice around line 56
// Method: <com.google.common.collect.EmptyContiguousSet: ContiguousSet intersection(ContiguousSet)>

    throw new NoSuchElementException();
  }

  @Override
  public int size() {
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
