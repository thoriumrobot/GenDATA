// Source-based slice around line 105
// Method: <com.google.common.collect.StandardTable: boolean containsRow(Object)>

    for (Map<C, V> map : backingMap.values()) {
      if (safeContainsKey(map, columnKey)) {
        return true;
      }
    }
    return false;
  }

  @Override
  public boolean containsRow(@Nullable Object rowKey) {
    return rowKey != null && safeContainsKey(backingMap, rowKey);
  }

  @Override
  public boolean containsValue(@Nullable Object value) {
    return value != null && super.containsValue(value);
  }

  @Override
  public @Nullable V get(@Nullable Object rowKey, @Nullable Object columnKey) {
