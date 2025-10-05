// Source-based slice around line 268
// Method: <com.google.common.collect.TreeRangeSet: void replaceRangeWithSameLowerBound(Range)>

        // { > }
        replaceRangeWithSameLowerBound(
            Range.create(rangeToRemove.upperBound, rangeBelowUb.upperBound));
      }
    }

    rangesByLowerBound.subMap(rangeToRemove.lowerBound, rangeToRemove.upperBound).clear();
  }

  private void replaceRangeWithSameLowerBound(Range<C> range) {
    if (range.isEmpty()) {
      rangesByLowerBound.remove(range.lowerBound);
    } else {
      rangesByLowerBound.put(range.lowerBound, range);
    }
  }

  @LazyInit private transient @Nullable RangeSet<C> complement;

  @Override
