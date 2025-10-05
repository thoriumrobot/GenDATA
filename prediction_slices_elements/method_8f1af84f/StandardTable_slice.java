// Source-based slice around line 136
// Method: <com.google.common.collect.StandardTable: void clear()>

    for (Map<C, V> map : backingMap.values()) {
      size += map.size();
    }
    return size;
  }

  // Mutators

  @Override
  public void clear() {
    backingMap.clear();
  }

  private Map<C, V> getOrCreate(R rowKey) {
    Map<C, V> map = backingMap.get(rowKey);
    if (map == null) {
      map = factory.get();
      backingMap.put(rowKey, map);
    }
    return map;
