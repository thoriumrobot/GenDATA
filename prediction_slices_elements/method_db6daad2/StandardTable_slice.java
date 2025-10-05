// Source-based slice around line 140
// Method: <com.google.common.collect.StandardTable: Map getOrCreate(R)>

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
  }

  @CanIgnoreReturnValue
  @Override
