// Source-based slice around line 276
// Method: com.google.common.collect.TreeRangeSet.complement


  private void replaceRangeWithSameLowerBound(Range<C> range) {
    if (range.isEmpty()) {
      rangesByLowerBound.remove(range.lowerBound);
    } else {
      rangesByLowerBound.put(range.lowerBound, range);
    }
  }

  @LazyInit private transient @Nullable RangeSet<C> complement;

  @Override
  public RangeSet<C> complement() {
    RangeSet<C> result = complement;
    return (result == null) ? complement = new Complement() : result;
  }

  @VisibleForTesting
  static final class RangesByUpperBound<C extends Comparable<?>>
      extends AbstractNavigableMap<Cut<C>, Range<C>> {
