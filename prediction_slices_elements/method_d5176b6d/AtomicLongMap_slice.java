// Source-based slice around line 310
// Method: <com.google.common.util.concurrent.AtomicLongMap: String toString()>

   *
   * <p>This method is not atomic: the map may not be empty after returning if there were concurrent
   * writes.
   */
  public void clear() {
    map.clear();
  }

  @Override
  public String toString() {
    return map.toString();
  }

  /**
   * If {@code key} is not already associated with a value or if {@code key} is associated with
   * zero, associate it with {@code newValue}. Returns the previous value associated with {@code
   * key}, or zero if there was no mapping for {@code key}.
   */
  long putIfAbsent(K key, long newValue) {
    AtomicBoolean noValue = new AtomicBoolean(false);
