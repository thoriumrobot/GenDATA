// Source-based slice around line 42
// Method: <com.google.common.cache.ForwardingLoadingCache: LoadingCache delegate()>

 */
@GwtIncompatible
public abstract class ForwardingLoadingCache<K, V> extends ForwardingCache<K, V>
    implements LoadingCache<K, V> {

  /** Constructor for use by subclasses. */
  protected ForwardingLoadingCache() {}

  @Override
  protected abstract LoadingCache<K, V> delegate();

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public V get(K key) throws ExecutionException {
    return delegate().get(key);
  }

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public V getUnchecked(K key) {
