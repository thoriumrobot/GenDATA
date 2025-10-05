// Source-based slice around line 62
// Method: com.google.common.util.concurrent.AtomicLongMap.map

 *
 * <p><b>Warning:</b> Unlike {@code Multiset}, entries whose values are zero are not automatically
 * removed from the map. Instead they must be removed manually with {@link #removeAllZeros}.
 *
 * @author Charles Fry
 * @since 11.0
 */
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

