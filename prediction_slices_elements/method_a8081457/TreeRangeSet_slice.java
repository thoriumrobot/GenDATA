// Source-based slice around line 56
// Method: <com.google.common.collect.TreeRangeSet: TreeRangeSet create(RangeSet)>


  @VisibleForTesting final NavigableMap<Cut<C>, Range<C>> rangesByLowerBound;

  /** Creates an empty {@code TreeRangeSet} instance. */
  public static <C extends Comparable<?>> TreeRangeSet<C> create() {
    return new TreeRangeSet<>(new TreeMap<Cut<C>, Range<C>>());
  }

  /** Returns a {@code TreeRangeSet} initialized with the ranges in the specified range set. */
  public static <C extends Comparable<?>> TreeRangeSet<C> create(RangeSet<C> rangeSet) {
    TreeRangeSet<C> result = create();
    result.addAll(rangeSet);
    return result;
  }

  /**
   * Returns a {@code TreeRangeSet} representing the union of the specified ranges.
   *
   * <p>This is the smallest {@code RangeSet} which encloses each of the specified ranges. An
   * element will be contained in this {@code RangeSet} if and only if it is contained in at least
