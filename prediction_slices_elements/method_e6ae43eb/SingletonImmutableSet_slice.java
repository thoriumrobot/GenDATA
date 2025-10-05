// Source-based slice around line 91
// Method: <com.google.common.collect.SingletonImmutableSet: Object writeReplace()>

  public String toString() {
    return '[' + element.toString() + ']';
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
