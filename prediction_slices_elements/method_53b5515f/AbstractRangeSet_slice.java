// Source-based slice around line 39
// Method: <com.google.common.collect.AbstractRangeSet: boolean isEmpty()>

  @Override
  public boolean contains(C value) {
    return rangeContaining(value) != null;
  }

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
