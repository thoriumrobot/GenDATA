// Source-based slice around line 50
// Method: <com.google.common.cache.AbstractCache: V get(K,Callable)>

public abstract class AbstractCache<K, V> implements Cache<K, V> {

  /** Constructor for use by subclasses. */
  protected AbstractCache() {}

  /**
   * @since 11.0
   */
  @Override
  public V get(K key, Callable<? extends V> valueLoader) throws ExecutionException {
    throw new UnsupportedOperationException();
  }

  /**
   * {@inheritDoc}
   *
   * <p>This implementation of {@code getAllPresent} lacks any insight into the internal cache data
   * structure, and is thus forced to return the query keys instead of the cached keys. This is only
   * possible with an unsafe cast which requires {@code keys} to actually be of type {@code K}.
   *
