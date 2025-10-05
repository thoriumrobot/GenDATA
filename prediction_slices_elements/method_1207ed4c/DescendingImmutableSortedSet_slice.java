// Source-based slice around line 125
// Method: <com.google.common.collect.DescendingImmutableSortedSet: Object writeReplace()>

  @Override
  boolean isPartialView() {
    return forward.isPartialView();
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
  }
}
