// Source-based slice around line 42
// Method: <com.google.common.cache.ForwardingCache: Cache delegate()>

 * @since 10.0
 */
@GwtIncompatible
public abstract class ForwardingCache<K, V> extends ForwardingObject implements Cache<K, V> {

  /** Constructor for use by subclasses. */
  protected ForwardingCache() {}

  @Override
  protected abstract Cache<K, V> delegate();

  /**
   * @since 11.0
   */
  @Override
  public @Nullable V getIfPresent(Object key) {
    return delegate().getIfPresent(key);
  }

  /**
