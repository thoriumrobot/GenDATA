// Source-based slice around line 76
// Method: <com.google.common.collect.SingletonImmutableTable: ImmutableCollection createValues()>

    return 1;
  }

  @Override
  ImmutableSet<Cell<R, C, V>> createCellSet() {
    return ImmutableSet.of(cellOf(singleRowKey, singleColumnKey, singleValue));
  }

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
