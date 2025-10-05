// Source-based slice around line 56
// Method: <com.google.common.collect.SingletonImmutableTable: ImmutableMap columnMap()>

  @Override
  public ImmutableMap<R, V> column(C columnKey) {
    checkNotNull(columnKey);
    return containsColumn(columnKey)
        ? ImmutableMap.of(singleRowKey, singleValue)
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
