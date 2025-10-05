// Source-based slice around line 120
// Method: <com.google.common.collect.StandardTable: boolean isEmpty()>

    return value != null && super.containsValue(value);
  }

  @Override
  public @Nullable V get(@Nullable Object rowKey, @Nullable Object columnKey) {
    return (rowKey == null || columnKey == null) ? null : super.get(rowKey, columnKey);
  }

  @Override
  public boolean isEmpty() {
    return backingMap.isEmpty();
  }

  @Override
  public int size() {
    int size = 0;
    for (Map<C, V> map : backingMap.values()) {
      size += map.size();
    }
    return size;
