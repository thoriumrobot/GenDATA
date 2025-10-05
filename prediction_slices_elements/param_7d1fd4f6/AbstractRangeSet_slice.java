// Source-based slice around line 69
// Method: <com.google.common.collect.AbstractRangeSet: void removeAll(RangeSet)>

    return enclosesAll(other.asRanges());
  }

  @Override
  public void addAll(RangeSet<C> other) {
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
