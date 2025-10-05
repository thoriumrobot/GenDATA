// Source-based slice around line 87
// Method: <com.google.common.collect.StandardTable: boolean contains(Object,Object)>


  StandardTable(Map<R, Map<C, V>> backingMap, Supplier<? extends Map<C, V>> factory) {
    this.backingMap = backingMap;
    this.factory = factory;
  }

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
