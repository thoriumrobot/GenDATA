// Source-based slice around line 763
// Method: <com.google.common.collect.ImmutableRangeSet: Builder builder()>

   * user-created objects that aren't accessible via this range set's methods. This is generally
   * used to determine whether {@code copyOf} implementations should make an explicit copy to avoid
   * memory leaks.
   */
  boolean isPartialView() {
    return ranges.isPartialView();
  }

  /** Returns a new builder for an immutable range set. */
  public static <C extends Comparable<?>> Builder<C> builder() {
    return new Builder<>();
  }

  /**
   * A builder for immutable range sets.
   *
   * @since 14.0
   */
  public static class Builder<C extends Comparable<?>> {
    private final List<Range<C>> ranges;
