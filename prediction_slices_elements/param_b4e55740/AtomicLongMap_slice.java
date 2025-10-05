// Source-based slice around line 134
// Method: <com.google.common.util.concurrent.AtomicLongMap: long getAndAdd(K,long)>

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
  }

  /**
   * Updates the value currently associated with {@code key} with the specified function, and
   * returns the new value. If there is not currently a value associated with {@code key}, the
   * function is applied to {@code 0L}.
   *
   * @since 21.0
   */
