// Source-based slice around line 256
// Method: <com.google.common.util.concurrent.AtomicLongMap: void removeAllZeros()>

    return remove(key, 0);
  }

  /**
   * Removes all mappings from this map whose values are zero.
   *
   * <p>This method is not atomic: the map may be visible in intermediate states, where some of the
   * zero values have been removed and others have not.
   */
  public void removeAllZeros() {
    map.values().removeIf(x -> x == 0);
  }

  /**
   * Returns the sum of all values in this map.
   *
   * <p>This method is not atomic: the sum may or may not include other concurrent operations.
   */
  public long sum() {
    return map.values().stream().mapToLong(Long::longValue).sum();
