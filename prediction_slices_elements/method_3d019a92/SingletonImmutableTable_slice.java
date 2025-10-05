// Source-based slice around line 61
// Method: <com.google.common.collect.SingletonImmutableTable: ImmutableMap rowMap()>

        : ImmutableMap.of();
  }

  @Override
  public ImmutableMap<C, Map<R, V>> columnMap() {
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
