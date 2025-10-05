// Source-based slice around line 227
// Method: <com.google.common.util.concurrent.AtomicLongMap: long remove(K)>

  public void putAll(Map<? extends K, ? extends Long> m) {
    m.forEach(this::put);
  }

  /**
   * Removes and returns the value associated with {@code key}. If {@code key} is not in the map,
   * this method has no effect and returns zero.
   */
  @CanIgnoreReturnValue
  public long remove(K key) {
    Long result = map.remove(key);
    return (result == null) ? 0L : result.longValue();
  }

  /**
   * If {@code (key, value)} is currently in the map, this method removes it and returns true;
   * otherwise, this method returns false.
   */
  boolean remove(K key, long value) {
    return map.remove(key, value);
