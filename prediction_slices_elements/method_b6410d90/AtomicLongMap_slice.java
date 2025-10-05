// Source-based slice around line 277
// Method: <com.google.common.util.concurrent.AtomicLongMap: Map createAsMap()>


  @LazyInit private transient @Nullable Map<K, Long> asMap;

  /** Returns a live, read-only view of the map backing this {@code AtomicLongMap}. */
  public Map<K, Long> asMap() {
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
