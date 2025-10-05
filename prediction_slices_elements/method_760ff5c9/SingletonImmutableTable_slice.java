// Source-based slice around line 66
// Method: <com.google.common.collect.SingletonImmutableTable: int size()>

    return ImmutableMap.of(singleColumnKey, (Map<R, V>) ImmutableMap.of(singleRowKey, singleValue));
  }

  @Override
  public ImmutableMap<R, Map<C, V>> rowMap() {
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
