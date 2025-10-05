// Source-based slice around line 184
// Method: <com.google.common.util.concurrent.AtomicLongMap: long accumulateAndGet(K,long,LongBinaryOperator)>

  /**
   * Updates the value currently associated with {@code key} by combining it with {@code x} via the
   * specified accumulator function, returning the new value. The previous value associated with
   * {@code key} (or zero, if there is none) is passed as the first argument to {@code
   * accumulatorFunction}, and {@code x} is passed as the second argument.
   *
   * @since 21.0
   */
  @CanIgnoreReturnValue
  public long accumulateAndGet(K key, long x, LongBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return updateAndGet(key, oldValue -> accumulatorFunction.applyAsLong(oldValue, x));
  }

  /**
   * Updates the value currently associated with {@code key} by combining it with {@code x} via the
   * specified accumulator function, returning the old value. The previous value associated with
   * {@code key} (or zero, if there is none) is passed as the first argument to {@code
   * accumulatorFunction}, and {@code x} is passed as the second argument.
   *
