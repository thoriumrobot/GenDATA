// Source-based slice around line 37
// Method: com.google.common.collect.ImmutableMapKeySet.map


/**
 * {@code keySet()} implementation for {@link ImmutableMap}.
 *
 * @author Jesse Wilson
 * @author Kevin Bourrillion
 */
@GwtCompatible
final class ImmutableMapKeySet<K, V> extends IndexedImmutableSet<K> {
  private final ImmutableMap<K, V> map;

  ImmutableMapKeySet(ImmutableMap<K, V> map) {
    this.map = map;
  }

  @Override
  public int size() {
    return map.size();
  }

