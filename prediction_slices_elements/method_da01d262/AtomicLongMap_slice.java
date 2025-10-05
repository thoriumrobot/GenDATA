// Source-based slice around line 92
// Method: <com.google.common.util.concurrent.AtomicLongMap: long incrementAndGet(K)>

   */
  public long get(K key) {
    return map.getOrDefault(key, 0L);
  }

  /**
   * Increments by one the value currently associated with {@code key}, and returns the new value.
   */
  @CanIgnoreReturnValue
  public long incrementAndGet(K key) {
    return addAndGet(key, 1);
  }

  /**
   * Decrements by one the value currently associated with {@code key}, and returns the new value.
   */
  @CanIgnoreReturnValue
  public long decrementAndGet(K key) {
    return addAndGet(key, -1);
  }
