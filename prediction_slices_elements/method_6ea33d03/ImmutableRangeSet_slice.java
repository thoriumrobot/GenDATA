// Source-based slice around line 108
// Method: <com.google.common.collect.ImmutableRangeSet: ImmutableRangeSet all()>

    } else if (range.equals(Range.all())) {
      return all();
    } else {
      return new ImmutableRangeSet<>(ImmutableList.of(range));
    }
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
