// Source-based slice around line 117
// Method: <com.google.common.util.concurrent.AtomicLongMap: long getAndIncrement(K)>

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

  /**
   * Decrements by one the value currently associated with {@code key}, and returns the old value.
   */
  @CanIgnoreReturnValue
  public long getAndDecrement(K key) {
    return getAndAdd(key, -1);
  }
