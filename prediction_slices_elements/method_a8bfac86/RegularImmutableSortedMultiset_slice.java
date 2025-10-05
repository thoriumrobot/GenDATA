// Source-based slice around line 139
// Method: <com.google.common.collect.RegularImmutableSortedMultiset: Object writeReplace()>

  @Override
  boolean isPartialView() {
    return offset > 0 || length < cumulativeCounts.length - 1;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible // serialization
  Object writeReplace() {
    return super.writeReplace();
  }
}
