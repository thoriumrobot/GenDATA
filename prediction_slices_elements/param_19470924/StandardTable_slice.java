// Source-based slice around line 160
// Method: <com.google.common.collect.StandardTable: V remove(Object,Object)>

  public @Nullable V put(R rowKey, C columnKey, V value) {
    checkNotNull(rowKey);
    checkNotNull(columnKey);
    checkNotNull(value);
    return getOrCreate(rowKey).put(columnKey, value);
  }

  @CanIgnoreReturnValue
  @Override
  public @Nullable V remove(@Nullable Object rowKey, @Nullable Object columnKey) {
    if ((rowKey == null) || (columnKey == null)) {
      return null;
    }
    Map<C, V> map = safeGet(backingMap, rowKey);
    if (map == null) {
      return null;
    }
    V value = map.remove(columnKey);
    if (map.isEmpty()) {
      backingMap.remove(rowKey);
