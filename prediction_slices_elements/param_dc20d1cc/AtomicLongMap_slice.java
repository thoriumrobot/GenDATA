// Source-based slice around line 236
// Method: <com.google.common.util.concurrent.AtomicLongMap: boolean remove(K,long)>

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
  }

  /**
   * Atomically remove {@code key} from the map iff its associated value is 0.
   *
   * @since 20.0
   */
  @CanIgnoreReturnValue
  public boolean removeIfZero(K key) {
