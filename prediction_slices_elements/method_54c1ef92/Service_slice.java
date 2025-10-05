// Source-based slice around line 172
// Method: <com.google.common.util.concurrent.Service: Throwable failureCause()>

  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  void awaitTerminated(long timeout, TimeUnit unit) throws TimeoutException;

  /**
   * Returns the {@link Throwable} that caused this service to fail.
   *
   * @throws IllegalStateException if this service's state isn't {@linkplain State#FAILED FAILED}.
   * @since 14.0
   */
  Throwable failureCause();

  /**
   * Registers a {@link Listener} to be {@linkplain Executor#execute executed} on the given
   * executor. The listener will have the corresponding transition method called whenever the
   * service changes state. The listener will not have previous state changes replayed, so it is
   * suggested that listeners are added before the service starts.
   *
   * <p>{@code addListener} guarantees execution ordering across calls to a given listener but not
   * across calls to multiple listeners. Specifically, a given listener will have its callbacks
   * invoked in the same order as the underlying service enters those states. Additionally, at most
