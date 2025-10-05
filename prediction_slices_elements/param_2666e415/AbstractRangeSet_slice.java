// Source-based slice around line 59
// Method: <com.google.common.collect.AbstractRangeSet: boolean enclosesAll(RangeSet)>

    throw new UnsupportedOperationException();
  }

  @Override
  public void clear() {
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
