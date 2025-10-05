// Source-based slice around line 113
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet copyOf(RangeSet)>

  }

  /** Returns an immutable range set containing the single range {@link Range#all()}. */
  @SuppressWarnings("unchecked")
  static <C extends Comparable> ImmutableRangeSet<C> all() {
    return (ImmutableRangeSet<C>) ALL;
  }

  /** Returns an immutable copy of the specified {@code RangeSet}. */
  public static <C extends Comparable> ImmutableRangeSet<C> copyOf(RangeSet<C> rangeSet) {
    checkNotNull(rangeSet);
    if (rangeSet.isEmpty()) {
      return of();
    } else if (rangeSet.encloses(Range.all())) {
      return all();
    }

    if (rangeSet instanceof ImmutableRangeSet) {
      ImmutableRangeSet<C> immutableRangeSet = (ImmutableRangeSet<C>) rangeSet;
      if (!immutableRangeSet.isPartialView()) {
