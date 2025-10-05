// Source-based slice around line 393
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet complement()>

    @SuppressWarnings("RedundantOverride")
    @Override
    @J2ktIncompatible // serialization
    Object writeReplace() {
      return super.writeReplace();
    }
  }

  @Override
  public ImmutableRangeSet<C> complement() {
    if (complement != null) {
      return complement;
    } else if (ranges.isEmpty()) {
      return all();
    } else if (ranges.size() == 1 && ranges.get(0).equals(Range.all())) {
      return of();
    } else {
      return lazyComplement();
    }
  }
