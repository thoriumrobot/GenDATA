// Source-based slice around line 150
// Method: <com.google.common.collect.TreeRangeSet: boolean encloses(Range)>

      return true;
    }
    Entry<Cut<C>, Range<C>> priorEntry = rangesByLowerBound.lowerEntry(range.lowerBound);
    return priorEntry != null
        && priorEntry.getValue().isConnected(range)
        && !priorEntry.getValue().intersection(range).isEmpty();
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
