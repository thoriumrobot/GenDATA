// Source-based slice around line 125
// Method: <com.google.common.collect.StandardTable: int size()>

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
  }

  // Mutators

  @Override
