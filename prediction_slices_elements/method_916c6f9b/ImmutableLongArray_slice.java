// Source-based slice around line 636
// Method: <com.google.common.primitives.ImmutableLongArray: Object readResolve()>


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
