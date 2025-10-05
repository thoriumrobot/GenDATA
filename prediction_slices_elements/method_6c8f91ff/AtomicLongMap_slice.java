// Source-based slice around line 69
// Method: <com.google.common.util.concurrent.AtomicLongMap: AtomicLongMap create()>

@GwtCompatible
public final class AtomicLongMap<K> implements Serializable {
  private final ConcurrentHashMap<K, Long> map;

  private AtomicLongMap(ConcurrentHashMap<K, Long> map) {
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

