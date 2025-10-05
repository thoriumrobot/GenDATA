// Source-based slice around line 224
// Method: <com.google.common.collect.RegularContiguousSet: boolean equals(Object)>


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
      RegularContiguousSet<?> that = (RegularContiguousSet<?>) object;
      if (this.domain.equals(that.domain)) {
        return this.first().equals(that.first()) && this.last().equals(that.last());
      }
    }
    return super.equals(object);
  }
