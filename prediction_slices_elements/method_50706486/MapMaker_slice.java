// Source-based slice around line 283
// Method: <com.google.common.collect.MapMaker: ConcurrentMap makeMap()>

   * instance, so it can be invoked again to create multiple independent maps.
   *
   * <p>The bulk operations {@code putAll}, {@code equals}, and {@code clear} are not guaranteed to
   * be performed atomically on the returned map. Additionally, {@code size} and {@code
   * containsValue} are implemented as bulk read operations, and thus may fail to observe concurrent
   * writes.
   *
   * @return a serializable concurrent map having the requested features
   */
  public <K, V> ConcurrentMap<K, V> makeMap() {
    if (!useCustomMap) {
      return new ConcurrentHashMap<>(getInitialCapacity(), 0.75f, getConcurrencyLevel());
    }
    return MapMakerInternalMap.create(this);
  }

  /**
   * Returns a string representation for this MapMaker instance. The exact form of the returned
   * string is not specified.
   */
