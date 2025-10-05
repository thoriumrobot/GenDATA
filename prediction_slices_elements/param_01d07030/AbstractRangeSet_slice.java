// Source-based slice around line 36
// Method: <com.google.common.collect.AbstractRangeSet: Range rangeContaining(C)>

abstract class AbstractRangeSet<C extends Comparable> implements RangeSet<C> {
  AbstractRangeSet() {}

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
