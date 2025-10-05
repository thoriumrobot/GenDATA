// Source-based slice around line 146
// Method: <com.google.common.util.concurrent.AtomicLongMap: long updateAndGet(K,LongUnaryOperator)>


  /**
   * Updates the value currently associated with {@code key} with the specified function, and
   * returns the new value. If there is not currently a value associated with {@code key}, the
   * function is applied to {@code 0L}.
   *
   * @since 21.0
   */
  @CanIgnoreReturnValue
  public long updateAndGet(K key, LongUnaryOperator updaterFunction) {
    checkNotNull(updaterFunction);
    Long result =
        map.compute(
            key, (k, value) -> updaterFunction.applyAsLong(value == null ? 0L : value.longValue()));
    return requireNonNull(result);
  }

  /**
   * Updates the value currently associated with {@code key} with the specified function, and
   * returns the old value. If there is not currently a value associated with {@code key}, the
