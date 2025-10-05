// Source-based slice around line 305
// Method: <com.google.common.util.concurrent.AtomicLongMap: void clear()>

    return map.isEmpty();
  }

  /**
   * Removes all of the mappings from this map. The map will be empty after this call returns.
   *
   * <p>This method is not atomic: the map may not be empty after returning if there were concurrent
   * writes.
   */
  public void clear() {
    map.clear();
  }

  @Override
  public String toString() {
    return map.toString();
  }

  /**
   * If {@code key} is not already associated with a value or if {@code key} is associated with
