// Source-based slice around line 630
// Method: <com.google.common.primitives.ImmutableIntArray: Object writeReplace()>

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
}
