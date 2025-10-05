// Source-based slice around line 632
// Method: <com.google.common.primitives.ImmutableLongArray: Object writeReplace()>

   */
  public ImmutableLongArray trimmed() {
    return isPartialView() ? new ImmutableLongArray(toArray()) : this;
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
