// Source-based slice around line 74
// Method: <com.google.common.collect.AbstractRangeSet: boolean intersects(Range)>

    addAll(other.asRanges());
  }

  @Override
  public void removeAll(RangeSet<C> other) {
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
