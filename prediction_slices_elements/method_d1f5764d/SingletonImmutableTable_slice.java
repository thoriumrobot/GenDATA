// Source-based slice around line 83
// Method: <com.google.common.collect.SingletonImmutableTable: Object writeReplace()>


  @Override
  ImmutableCollection<V> createValues() {
    return ImmutableSet.of(singleValue);
  }

  @Override
  @J2ktIncompatible
  @GwtIncompatible
    Object writeReplace() {
    return SerializedForm.create(this, new int[] {0}, new int[] {0});
  }
}
