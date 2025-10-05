// Source-based slice around line 71
// Method: <com.google.common.collect.TreeRangeSet: TreeRangeSet create(Iterable)>

  /**
   * Returns a {@code TreeRangeSet} representing the union of the specified ranges.
   *
   * <p>This is the smallest {@code RangeSet} which encloses each of the specified ranges. An
   * element will be contained in this {@code RangeSet} if and only if it is contained in at least
   * one {@code Range} in {@code ranges}.
   *
   * @since 21.0
   */
  public static <C extends Comparable<?>> TreeRangeSet<C> create(Iterable<Range<C>> ranges) {
    TreeRangeSet<C> result = create();
    result.addAll(ranges);
    return result;
  }

  private TreeRangeSet(NavigableMap<Cut<C>, Range<C>> rangesByLowerCut) {
    this.rangesByLowerBound = rangesByLowerCut;
  }

  @LazyInit private transient @Nullable Set<Range<C>> asRanges;
