// Source-based slice around line 86
// Method: <com.google.common.collect.SingletonImmutableList: Object writeReplace()>

  boolean isPartialView() {
    return false;
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
