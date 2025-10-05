// Source-based slice around line 246
// Method: <com.google.common.util.concurrent.AtomicLongMap: boolean removeIfZero(K)>

    return map.remove(key, value);
  }

  /**
   * Atomically remove {@code key} from the map iff its associated value is 0.
   *
   * @since 20.0
   */
  @CanIgnoreReturnValue
  public boolean removeIfZero(K key) {
    return remove(key, 0);
  }

  /**
   * Removes all mappings from this map whose values are zero.
   *
   * <p>This method is not atomic: the map may be visible in intermediate states, where some of the
   * zero values have been removed and others have not.
   */
  public void removeAllZeros() {
