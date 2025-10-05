// Source-based slice around line 159
// Method: <com.google.common.collect.RangeSet: RangeSet subRangeSet(Range)>

  RangeSet<C> complement();

  /**
   * Returns a view of the intersection of this {@code RangeSet} with the specified range.
   *
   * <p>The returned view supports all optional operations supported by this {@code RangeSet}, with
   * the caveat that an {@link IllegalArgumentException} is thrown on an attempt to {@linkplain
   * #add(Range) add} any range not {@linkplain Range#encloses(Range) enclosed} by {@code view}.
   */
  RangeSet<C> subRangeSet(Range<C> view);

  // Modification

  /**
   * Adds the specified range to this {@code RangeSet} (optional operation). That is, for equal
   * range sets a and b, the result of {@code a.add(range)} is that {@code a} will be the minimal
   * range set for which both {@code a.enclosesAll(b)} and {@code a.encloses(range)}.
   *
   * <p>Note that {@code range} will be {@linkplain Range#span(Range) coalesced} with any ranges in
   * the range set that are {@linkplain Range#isConnected(Range) connected} with it. Moreover, if
