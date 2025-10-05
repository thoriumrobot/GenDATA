// Source-based slice around line 60
// Method: <com.google.common.collect.RangeSet: boolean contains(C)>

@SuppressWarnings("rawtypes") // https://github.com/google/guava/issues/989
@DoNotMock("Use ImmutableRangeSet or TreeRangeSet")
@GwtIncompatible
public interface RangeSet<C extends Comparable> {
  // TODO(lowasser): consider adding default implementations of some of these methods

  // Query methods

  /** Determines whether any of this range set's member ranges contains {@code value}. */
  boolean contains(C value);

  /**
   * Returns the unique range from this range set that {@linkplain Range#contains contains} {@code
   * value}, or {@code null} if this range set does not contain {@code value}.
   */
  @Nullable Range<C> rangeContaining(C value);

  /**
   * Returns {@code true} if there exists a non-empty range enclosed by both a member range in this
   * range set and the specified range. This is equivalent to calling {@code
