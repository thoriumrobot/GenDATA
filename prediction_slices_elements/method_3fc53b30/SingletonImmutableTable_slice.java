// Source-based slice around line 71
// Method: <com.google.common.collect.SingletonImmutableTable: ImmutableSet createCellSet()>

    return ImmutableMap.of(singleRowKey, (Map<C, V>) ImmutableMap.of(singleColumnKey, singleValue));
  }

  @Override
  public int size() {
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
