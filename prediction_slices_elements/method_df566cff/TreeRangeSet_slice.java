// Source-based slice around line 859
// Method: <com.google.common.collect.TreeRangeSet: RangeSet subRangeSet(Range)>

    }

    @Override
    public int size() {
      return Iterators.size(entryIterator());
    }
  }

  @Override
  public RangeSet<C> subRangeSet(Range<C> view) {
    return view.equals(Range.all()) ? this : new SubRangeSet(view);
  }

  private final class SubRangeSet extends TreeRangeSet<C> {
    private final Range<C> restriction;

    SubRangeSet(Range<C> restriction) {
      super(
          new SubRangeSetRangesByLowerBound<C>(
              Range.all(), restriction, TreeRangeSet.this.rangesByLowerBound));
