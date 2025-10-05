// Source-based slice around line 109
// Method: <com.google.common.util.concurrent.AtomicLongMap: long addAndGet(K,long)>

  public long decrementAndGet(K key) {
    return addAndGet(key, -1);
  }

  /**
   * Adds {@code delta} to the value currently associated with {@code key}, and returns the new
   * value.
   */
  @CanIgnoreReturnValue
  public long addAndGet(K key, long delta) {
    return accumulateAndGet(key, delta, Long::sum);
  }

  /**
   * Increments by one the value currently associated with {@code key}, and returns the old value.
   */
  @CanIgnoreReturnValue
  public long getAndIncrement(K key) {
    return getAndAdd(key, 1);
  }
