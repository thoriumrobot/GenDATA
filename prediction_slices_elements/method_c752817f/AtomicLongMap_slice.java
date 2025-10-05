// Source-based slice around line 272
// Method: <com.google.common.util.concurrent.AtomicLongMap: Map asMap()>

   * <p>This method is not atomic: the sum may or may not include other concurrent operations.
   */
  public long sum() {
    return map.values().stream().mapToLong(Long::longValue).sum();
  }

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
