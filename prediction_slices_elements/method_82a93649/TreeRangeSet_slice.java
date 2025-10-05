// Source-based slice around line 279
// Method: <com.google.common.collect.TreeRangeSet: RangeSet complement()>

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
    private final NavigableMap<Cut<C>, Range<C>> rangesByLowerBound;

    /**
