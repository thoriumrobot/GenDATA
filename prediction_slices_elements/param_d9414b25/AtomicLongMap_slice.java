// Source-based slice around line 100
// Method: <com.google.common.util.concurrent.AtomicLongMap: long decrementAndGet(K)>

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

  /**
   * Adds {@code delta} to the value currently associated with {@code key}, and returns the new
   * value.
   */
  @CanIgnoreReturnValue
  public long addAndGet(K key, long delta) {
    return accumulateAndGet(key, delta, Long::sum);
