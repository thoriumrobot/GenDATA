// Source-based slice around line 295
// Method: <com.google.common.util.concurrent.AtomicLongMap: boolean isEmpty()>

  /**
   * Returns the number of key-value mappings in this map. If the map contains more than {@code
   * Integer.MAX_VALUE} elements, returns {@code Integer.MAX_VALUE}.
   */
  public int size() {
    return map.size();
  }

  /** Returns {@code true} if this map contains no key-value mappings. */
  public boolean isEmpty() {
    return map.isEmpty();
  }

  /**
   * Removes all of the mappings from this map. The map will be empty after this call returns.
   *
   * <p>This method is not atomic: the map may not be empty after returning if there were concurrent
   * writes.
   */
  public void clear() {
