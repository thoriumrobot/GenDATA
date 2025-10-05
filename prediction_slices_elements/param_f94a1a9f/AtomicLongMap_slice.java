// Source-based slice around line 208
// Method: <com.google.common.util.concurrent.AtomicLongMap: long put(K,long)>

    checkNotNull(accumulatorFunction);
    return getAndUpdate(key, oldValue -> accumulatorFunction.applyAsLong(oldValue, x));
  }

  /**
   * Associates {@code newValue} with {@code key} in this map, and returns the value previously
   * associated with {@code key}, or zero if there was no such value.
   */
  @CanIgnoreReturnValue
  public long put(K key, long newValue) {
    return getAndUpdate(key, x -> newValue);
  }

  /**
   * Copies all of the mappings from the specified map to this map. The effect of this call is
   * equivalent to that of calling {@code put(k, v)} on this map once for each mapping from key
   * {@code k} to value {@code v} in the specified map. The behavior of this operation is undefined
   * if the specified map is modified while the operation is in progress.
   */
  public void putAll(Map<? extends K, ? extends Long> m) {
