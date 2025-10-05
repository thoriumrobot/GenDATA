// Source-based slice around line 81
// Method: com.google.common.collect.TreeRangeSet.asRanges

    TreeRangeSet<C> result = create();
    result.addAll(ranges);
    return result;
  }

  private TreeRangeSet(NavigableMap<Cut<C>, Range<C>> rangesByLowerCut) {
    this.rangesByLowerBound = rangesByLowerCut;
  }

  @LazyInit private transient @Nullable Set<Range<C>> asRanges;
  @LazyInit private transient @Nullable Set<Range<C>> asDescendingSetOfRanges;

  @Override
  public Set<Range<C>> asRanges() {
    Set<Range<C>> result = asRanges;
    return (result == null) ? asRanges = new AsRanges(rangesByLowerBound.values()) : result;
  }

  @Override
  public Set<Range<C>> asDescendingSetOfRanges() {
