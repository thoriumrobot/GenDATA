// Source-based slice around line 50
// Method: <com.google.common.cache.AbstractLoadingCache: V getUnchecked(K)>

@GwtIncompatible
public abstract class AbstractLoadingCache<K, V> extends AbstractCache<K, V>
    implements LoadingCache<K, V> {

  /** Constructor for use by subclasses. */
  protected AbstractLoadingCache() {}

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this?
  @Override
  public V getUnchecked(K key) {
    try {
      return get(key);
    } catch (ExecutionException e) {
      throw new UncheckedExecutionException(e.getCause());
    }
  }

  @Override
  public ImmutableMap<K, V> getAll(Iterable<? extends K> keys) throws ExecutionException {
    Map<K, V> result = new LinkedHashMap<>();
