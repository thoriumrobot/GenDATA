// Source-based slice around line 226
// Method: <com.google.common.collect.TreeRangeSet: void remove(Range)>

    }

    // Remove ranges which are strictly enclosed.
    rangesByLowerBound.subMap(lbToAdd, ubToAdd).clear();

    replaceRangeWithSameLowerBound(Range.create(lbToAdd, ubToAdd));
  }

  @Override
  public void remove(Range<C> rangeToRemove) {
    checkNotNull(rangeToRemove);

    if (rangeToRemove.isEmpty()) {
      return;
    }

    // We will use { } to illustrate ranges currently in the range set, and < >
    // to illustrate rangeToRemove.

    Entry<Cut<C>, Range<C>> entryBelowLb = rangesByLowerBound.lowerEntry(rangeToRemove.lowerBound);
