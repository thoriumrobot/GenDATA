// Source-based slice around line 156
// Method: <com.google.common.collect.TreeRangeSet: Range rangeEnclosing(Range)>

  }

  @Override
  public boolean encloses(Range<C> range) {
    checkNotNull(range);
    Entry<Cut<C>, Range<C>> floorEntry = rangesByLowerBound.floorEntry(range.lowerBound);
    return floorEntry != null && floorEntry.getValue().encloses(range);
  }

  private @Nullable Range<C> rangeEnclosing(Range<C> range) {
    checkNotNull(range);
    Entry<Cut<C>, Range<C>> floorEntry = rangesByLowerBound.floorEntry(range.lowerBound);
    return (floorEntry != null && floorEntry.getValue().encloses(range))
        ? floorEntry.getValue()
        : null;
  }

  @Override
  public Range<C> span() {
    Entry<Cut<C>, Range<C>> firstEntry = rangesByLowerBound.firstEntry();
