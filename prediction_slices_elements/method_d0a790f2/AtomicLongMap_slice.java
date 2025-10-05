// Source-based slice around line 84
// Method: <com.google.common.util.concurrent.AtomicLongMap: long get(K)>

    AtomicLongMap<K> result = create();
    result.putAll(m);
    return result;
  }

  /**
   * Returns the value associated with {@code key}, or zero if there is no value associated with
   * {@code key}.
   */
  public long get(K key) {
    return map.getOrDefault(key, 0L);
  }

  /**
   * Increments by one the value currently associated with {@code key}, and returns the new value.
   */
  @CanIgnoreReturnValue
  public long incrementAndGet(K key) {
    return addAndGet(key, 1);
  }
