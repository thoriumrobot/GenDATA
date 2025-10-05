// Source-based slice around line 290
// Method: <com.google.common.util.concurrent.AtomicLongMap: int size()>

  /** Returns true if this map contains a mapping for the specified key. */
  public boolean containsKey(Object key) {
    return map.containsKey(key);
  }

  /**
   * Returns the number of key-value mappings in this map. If the map contains more than {@code
   * Integer.MAX_VALUE} elements, returns {@code Integer.MAX_VALUE}.
   */
  public int size() {
    return map.size();
  }

  /** Returns {@code true} if this map contains no key-value mappings. */
  public boolean isEmpty() {
    return map.isEmpty();
  }

  /**
   * Removes all of the mappings from this map. The map will be empty after this call returns.
