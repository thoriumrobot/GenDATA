// Source-based slice around line 129
// Method: <com.google.common.collect.EmptyContiguousSet: boolean equals(Object)>

    return ImmutableList.of();
  }

  @Override
  public String toString() {
    return "[]";
  }

  @Override
  public boolean equals(@Nullable Object object) {
    if (object instanceof Set) {
      Set<?> that = (Set<?>) object;
      return that.isEmpty();
    }
    return false;
  }

  @GwtIncompatible // not used in GWT
  @Override
  boolean isHashCodeFast() {
