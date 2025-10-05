// Source-based slice around line 176
// Method: <com.google.common.collect.StandardTable: Map removeColumn(Object)>

    }
    V value = map.remove(columnKey);
    if (map.isEmpty()) {
      backingMap.remove(rowKey);
    }
    return value;
  }

  @CanIgnoreReturnValue
  private Map<R, V> removeColumn(@Nullable Object column) {
    Map<R, V> output = new LinkedHashMap<>();
    Iterator<Entry<R, Map<C, V>>> iterator = backingMap.entrySet().iterator();
    while (iterator.hasNext()) {
      Entry<R, Map<C, V>> entry = iterator.next();
      V value = entry.getValue().remove(column);
      if (value != null) {
        output.put(entry.getKey(), value);
        if (entry.getValue().isEmpty()) {
          iterator.remove();
        }
