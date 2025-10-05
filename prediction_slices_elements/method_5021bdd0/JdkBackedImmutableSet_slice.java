// Source-based slice around line 63
// Method: <com.google.common.collect.JdkBackedImmutableSet: Object writeReplace()>

  public int size() {
    return delegateList.size();
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
