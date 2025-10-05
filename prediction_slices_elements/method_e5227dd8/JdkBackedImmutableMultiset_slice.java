// Source-based slice around line 90
// Method: <com.google.common.collect.JdkBackedImmutableMultiset: int size()>

    return entries.get(index);
  }

  @Override
  boolean isPartialView() {
    return false;
  }

  @Override
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
