// Source-based slice around line 626
// Method: <com.google.common.primitives.ImmutableIntArray: boolean isPartialView()>

   * Returns an immutable array containing the same values as {@code this} array. This is logically
   * a no-op, and in some circumstances {@code this} itself is returned. However, if this instance
   * is a {@link #subArray} view of a larger array, this method will copy only the appropriate range
   * of values, resulting in an equivalent array with a smaller memory footprint.
   */
  public ImmutableIntArray trimmed() {
    return isPartialView() ? new ImmutableIntArray(toArray()) : this;
  }

  private boolean isPartialView() {
    return start > 0 || end < array.length;
  }

  Object writeReplace() {
    return trimmed();
  }

  Object readResolve() {
    return isEmpty() ? EMPTY : this;
  }
