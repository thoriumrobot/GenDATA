// Source-based slice around line 44
// Method: <com.google.common.collect.AbstractRangeSet: void add(Range)>

  @Override
  public abstract @Nullable Range<C> rangeContaining(C value);

  @Override
  public boolean isEmpty() {
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
