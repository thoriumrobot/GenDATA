// Source-based slice around line 74
// Method: <com.google.common.util.concurrent.AtomicLongMap: AtomicLongMap create(Map)>

    this.map = checkNotNull(map);
  }

  /** Creates an {@code AtomicLongMap}. */
  public static <K> AtomicLongMap<K> create() {
    return new AtomicLongMap<>(new ConcurrentHashMap<>());
  }

  /** Creates an {@code AtomicLongMap} with the same mappings as the specified {@code Map}. */
  public static <K> AtomicLongMap<K> create(Map<? extends K, ? extends Long> m) {
    AtomicLongMap<K> result = create();
    result.putAll(m);
    return result;
  }

  /**
   * Returns the value associated with {@code key}, or zero if there is no value associated with
   * {@code key}.
   */
  public long get(K key) {
