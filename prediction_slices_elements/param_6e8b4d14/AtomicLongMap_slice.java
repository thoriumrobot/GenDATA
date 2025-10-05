// Source-based slice around line 218
// Method: <com.google.common.util.concurrent.AtomicLongMap: void putAll(Map)>

    return getAndUpdate(key, x -> newValue);
  }

  /**
   * Copies all of the mappings from the specified map to this map. The effect of this call is
   * equivalent to that of calling {@code put(k, v)} on this map once for each mapping from key
   * {@code k} to value {@code v} in the specified map. The behavior of this operation is undefined
   * if the specified map is modified while the operation is in progress.
   */
  public void putAll(Map<? extends K, ? extends Long> m) {
    m.forEach(this::put);
  }

  /**
   * Removes and returns the value associated with {@code key}. If {@code key} is not in the map,
   * this method has no effect and returns zero.
   */
  @CanIgnoreReturnValue
  public long remove(K key) {
    Long result = map.remove(key);
