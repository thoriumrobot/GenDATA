// Source-based slice around line 77
// Method: <com.google.common.collect.SingletonImmutableList: boolean isPartialView()>

    return (fromIndex == toIndex) ? ImmutableList.of() : this;
  }

  @Override
  public String toString() {
    return '[' + element.toString() + ']';
  }

  @Override
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
