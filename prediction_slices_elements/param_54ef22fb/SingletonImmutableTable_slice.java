// Source-based slice around line 48
// Method: <com.google.common.collect.SingletonImmutableTable: ImmutableMap column(C)>

    this.singleColumnKey = checkNotNull(columnKey);
    this.singleValue = checkNotNull(value);
  }

  SingletonImmutableTable(Cell<R, C, V> cell) {
    this(cell.getRowKey(), cell.getColumnKey(), cell.getValue());
  }

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
