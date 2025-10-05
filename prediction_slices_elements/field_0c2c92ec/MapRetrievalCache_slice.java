// Source-based slice around line 33
// Method: com.google.common.graph.MapRetrievalCache.cacheEntry2

/**
 * A {@link MapIteratorCache} that adds additional caching. In addition to the caching provided by
 * {@link MapIteratorCache}, this structure caches values for the two most recently retrieved keys.
 *
 * @author James Sexton
 */
final class MapRetrievalCache<K, V> extends MapIteratorCache<K, V> {
  // See the note about volatile in the superclass.
  private transient volatile @Nullable CacheEntry<K, V> cacheEntry1;
  private transient volatile @Nullable CacheEntry<K, V> cacheEntry2;

  MapRetrievalCache(Map<K, V> backingMap) {
    super(backingMap);
  }

  @SuppressWarnings("unchecked") // Safe because we only cast if key is found in map.
  @Override
  @Nullable V get(Object key) {
    checkNotNull(key);
    V value = getIfCached(key);
