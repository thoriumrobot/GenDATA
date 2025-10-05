// Source-based slice around line 136
// Method: <com.google.common.collect.RegularImmutableSet: Object writeReplace()>

  boolean isHashCodeFast() {
    return true;
  }

  // redeclare to help optimizers with b/310253115
  @SuppressWarnings("RedundantOverride")
  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return super.writeReplace();
  }
}
