// Source-based slice around line 640
// Method: <com.google.common.primitives.ImmutableDoubleArray: Object writeReplace()>

   */
  public ImmutableDoubleArray trimmed() {
    return isPartialView() ? new ImmutableDoubleArray(toArray()) : this;
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
