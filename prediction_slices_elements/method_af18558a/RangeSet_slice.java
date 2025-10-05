// Source-based slice around line 150
// Method: <com.google.common.collect.RangeSet: RangeSet complement()>

   */
  Set<Range<C>> asDescendingSetOfRanges();

  /**
   * Returns a view of the complement of this {@code RangeSet}.
   *
   * <p>The returned view supports the {@link #add} operation if this {@code RangeSet} supports
   * {@link #remove}, and vice versa.
   */
  RangeSet<C> complement();

  /**
   * Returns a view of the intersection of this {@code RangeSet} with the specified range.
   *
   * <p>The returned view supports all optional operations supported by this {@code RangeSet}, with
   * the caveat that an {@link IllegalArgumentException} is thrown on an attempt to {@linkplain
   * #add(Range) add} any range not {@linkplain Range#encloses(Range) enclosed} by {@code view}.
   */
  RangeSet<C> subRangeSet(Range<C> view);

