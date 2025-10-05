// Source-based slice around line 274
// Method: <com.google.common.collect.ContiguousSet: Object writeReplace()>

  public static <E> ImmutableSortedSet.Builder<E> builder() {
    throw new UnsupportedOperationException();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @J2ktIncompatible // serialization
  @Override
  @GwtIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
  }
}
