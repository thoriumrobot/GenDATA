// Source-based slice around line 319
// Method: <com.google.common.util.concurrent.AtomicLongMap: long putIfAbsent(K,long)>

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
    Long result =
        map.compute(
            key,
            (k, oldValue) -> {
              if (oldValue == null || oldValue == 0) {
                noValue.set(true);
                return newValue;
              } else {
                return oldValue;
