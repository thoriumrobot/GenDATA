// Source-based slice around line 79
// Method: <com.google.common.collect.AbstractRangeSet: boolean encloses(Range)>

    removeAll(other.asRanges());
  }

  @Override
  public boolean intersects(Range<C> otherRange) {
    return !subRangeSet(otherRange).isEmpty();
  }

  @Override
  public abstract boolean encloses(Range<C> otherRange);

  @Override
  public boolean equals(@Nullable Object obj) {
    if (obj == this) {
      return true;
    } else if (obj instanceof RangeSet) {
      RangeSet<?> other = (RangeSet<?>) obj;
      return this.asRanges().equals(other.asRanges());
    }
    return false;
