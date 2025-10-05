// Source-based slice around line 110
// Method: <com.google.common.collect.StandardTable: boolean containsValue(Object)>

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
    return (rowKey == null || columnKey == null) ? null : super.get(rowKey, columnKey);
  }

  @Override
  public boolean isEmpty() {
