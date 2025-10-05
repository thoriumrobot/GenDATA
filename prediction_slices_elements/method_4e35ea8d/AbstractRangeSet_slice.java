// Source-based slice around line 54
// Method: <com.google.common.collect.AbstractRangeSet: void clear()>

    throw new UnsupportedOperationException();
  }

  @Override
  public void remove(Range<C> range) {
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
