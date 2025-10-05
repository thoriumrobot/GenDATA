// Source-based slice around line 198
// Method: <com.google.common.util.concurrent.AtomicLongMap: long getAndAccumulate(K,long,LongBinaryOperator)>

  /**
   * Updates the value currently associated with {@code key} by combining it with {@code x} via the
   * specified accumulator function, returning the old value. The previous value associated with
   * {@code key} (or zero, if there is none) is passed as the first argument to {@code
   * accumulatorFunction}, and {@code x} is passed as the second argument.
   *
   * @since 21.0
   */
  @CanIgnoreReturnValue
  public long getAndAccumulate(K key, long x, LongBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return getAndUpdate(key, oldValue -> accumulatorFunction.applyAsLong(oldValue, x));
  }

  /**
   * Associates {@code newValue} with {@code key} in this map, and returns the value previously
   * associated with {@code key}, or zero if there was no such value.
   */
  @CanIgnoreReturnValue
  public long put(K key, long newValue) {
