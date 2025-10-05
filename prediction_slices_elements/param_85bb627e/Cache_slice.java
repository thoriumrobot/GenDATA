// Source-based slice around line 54
// Method: <com.google.common.cache.Cache: V getIfPresent(Object)>

public interface Cache<K, V> {

  /**
   * Returns the value associated with {@code key} in this cache, or {@code null} if there is no
   * cached value for {@code key}.
   *
   * @since 11.0
   */
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this?
  @Nullable V getIfPresent(@CompatibleWith("K") Object key);

  /**
   * Returns the value associated with {@code key} in this cache, obtaining that value from {@code
   * loader} if necessary. The method improves upon the conventional "if cached, return; otherwise
   * create, cache and return" pattern. For further improvements, use {@link LoadingCache} and its
   * {@link LoadingCache#get(Object) get(K)} method instead of this one.
   *
   * <p>Among the improvements that this method and {@code LoadingCache.get(K)} both provide are:
   *
   * <ul>
