// Source-based slice around line 49
// Method: <com.google.common.collect.AbstractRangeSet: void remove(Range)>

    return asRanges().isEmpty();
  }

  @Override
  public void add(Range<C> range) {
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
