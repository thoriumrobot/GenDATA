// Source-based slice around line 282
// Method: <com.google.common.util.concurrent.AtomicLongMap: boolean containsKey(Object)>

    Map<K, Long> result = asMap;
    return (result == null) ? asMap = createAsMap() : result;
  }

  private Map<K, Long> createAsMap() {
    return Collections.unmodifiableMap(map);
  }

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
