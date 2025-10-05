// Source-based slice around line 125
// Method: <com.google.common.util.concurrent.AtomicLongMap: long getAndDecrement(K)>

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

  /**
   * Adds {@code delta} to the value currently associated with {@code key}, and returns the old
   * value.
   */
  @CanIgnoreReturnValue
  public long getAndAdd(K key, long delta) {
    return getAndAccumulate(key, delta, Long::sum);
