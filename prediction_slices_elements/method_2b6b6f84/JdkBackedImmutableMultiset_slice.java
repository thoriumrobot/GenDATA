// Source-based slice around line 99
// Method: <com.google.common.collect.JdkBackedImmutableMultiset: Object writeReplace()>

  public int size() {
    return Ints.saturatedCast(size);
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
