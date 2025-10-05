// Source-based slice around line 91
// Method: <com.google.common.collect.TreeRangeSet: Set asDescendingSetOfRanges()>

  @LazyInit private transient @Nullable Set<Range<C>> asDescendingSetOfRanges;

  @Override
  public Set<Range<C>> asRanges() {
    Set<Range<C>> result = asRanges;
    return (result == null) ? asRanges = new AsRanges(rangesByLowerBound.values()) : result;
  }

  @Override
  public Set<Range<C>> asDescendingSetOfRanges() {
    Set<Range<C>> result = asDescendingSetOfRanges;
    return (result == null)
        ? asDescendingSetOfRanges = new AsRanges(rangesByLowerBound.descendingMap().values())
        : result;
  }

  final class AsRanges extends ForwardingCollection<Range<C>> implements Set<Range<C>> {

    final Collection<Range<C>> delegate;

