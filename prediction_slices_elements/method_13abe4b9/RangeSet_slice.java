// Source-based slice around line 104
// Method: <com.google.common.collect.RangeSet: boolean enclosesAll(Iterable)>

   * Returns {@code true} if for each range in {@code other} there exists a member range in this
   * range set which {@linkplain Range#encloses encloses} it. Returns {@code true} if {@code other}
   * is empty.
   *
   * <p>This is equivalent to checking if this range set {@link #encloses} each range in {@code
   * other}.
   *
   * @since 21.0
   */
  default boolean enclosesAll(Iterable<Range<C>> other) {
    for (Range<C> range : other) {
      if (!encloses(range)) {
        return false;
      }
    }
    return true;
  }

  /** Returns {@code true} if this range set contains no ranges. */
  boolean isEmpty();
