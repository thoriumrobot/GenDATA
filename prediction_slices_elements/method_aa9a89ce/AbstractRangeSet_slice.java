// Source-based slice around line 64
// Method: <com.google.common.collect.AbstractRangeSet: void addAll(RangeSet)>

    remove(Range.all());
  }

  @Override
  public boolean enclosesAll(RangeSet<C> other) {
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
