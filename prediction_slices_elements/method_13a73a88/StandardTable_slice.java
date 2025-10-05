// Source-based slice around line 92
// Method: <com.google.common.collect.StandardTable: boolean containsColumn(Object)>


  // Accessors

  @Override
  public boolean contains(@Nullable Object rowKey, @Nullable Object columnKey) {
    return rowKey != null && columnKey != null && super.contains(rowKey, columnKey);
  }

  @Override
  public boolean containsColumn(@Nullable Object columnKey) {
    if (columnKey == null) {
      return false;
    }
    for (Map<C, V> map : backingMap.values()) {
      if (safeContainsKey(map, columnKey)) {
        return true;
      }
    }
    return false;
  }
