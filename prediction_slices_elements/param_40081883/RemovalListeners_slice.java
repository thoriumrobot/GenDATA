// Source-based slice around line 41
// Method: <com.google.common.cache.RemovalListeners: RemovalListener asynchronous(RemovalListener,Executor)>


  /**
   * Returns a {@code RemovalListener} which processes all eviction notifications using {@code
   * executor}.
   *
   * @param listener the backing listener
   * @param executor the executor with which removal notifications are asynchronously executed
   */
  public static <K, V> RemovalListener<K, V> asynchronous(
      RemovalListener<K, V> listener, Executor executor) {
    checkNotNull(listener);
    checkNotNull(executor);
    return (RemovalNotification<K, V> notification) ->
        executor.execute(() -> listener.onRemoval(notification));
  }
}
