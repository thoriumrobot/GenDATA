// Source-based slice around line 162
// Method: <com.google.common.util.concurrent.AtomicLongMap: long getAndUpdate(K,LongUnaryOperator)>


  /**
   * Updates the value currently associated with {@code key} with the specified function, and
   * returns the old value. If there is not currently a value associated with {@code key}, the
   * function is applied to {@code 0L}.
   *
   * @since 21.0
   */
  @CanIgnoreReturnValue
  public long getAndUpdate(K key, LongUnaryOperator updaterFunction) {
    checkNotNull(updaterFunction);
    AtomicLong holder = new AtomicLong();
    map.compute(
        key,
        (k, value) -> {
          long oldValue = (value == null) ? 0L : value.longValue();
          holder.set(oldValue);
          return updaterFunction.applyAsLong(oldValue);
        });
    return holder.get();
