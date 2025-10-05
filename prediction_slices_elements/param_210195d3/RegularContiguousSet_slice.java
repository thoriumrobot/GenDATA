// Source-based slice around line 217
// Method: <com.google.common.collect.RegularContiguousSet: Range range(BoundType,BoundType)>

    }
  }

  @Override
  public Range<C> range() {
    return range(CLOSED, CLOSED);
  }

  @Override
  public Range<C> range(BoundType lowerBoundType, BoundType upperBoundType) {
    return Range.create(
        range.lowerBound.withLowerBoundType(lowerBoundType, domain),
        range.upperBound.withUpperBoundType(upperBoundType, domain));
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    } else if (object instanceof RegularContiguousSet) {
