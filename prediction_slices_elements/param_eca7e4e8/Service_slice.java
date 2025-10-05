// Source-based slice around line 164
// Method: <com.google.common.util.concurrent.Service: void awaitTerminated(long,TimeUnit)>

   * terminated} or {@link Service.State#FAILED failed}) for no more than the given time.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
   * @throws TimeoutException if the service has not reached the given state within the deadline
   * @throws IllegalStateException if the service {@linkplain State#FAILED fails}.
   * @since 15.0
   */
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
