// Source-based slice around line 138
// Method: <com.google.common.util.concurrent.Service: void awaitTerminated()>

  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  void awaitRunning(long timeout, TimeUnit unit) throws TimeoutException;

  /**
   * Waits for the {@link Service} to reach the {@linkplain State#TERMINATED terminated state}.
   *
   * @throws IllegalStateException if the service {@linkplain State#FAILED fails}.
   * @since 15.0
   */
  void awaitTerminated();

  /**
   * Waits for the {@link Service} to reach a terminal state (either {@link Service.State#TERMINATED
   * terminated} or {@link Service.State#FAILED failed}) for no more than the given time.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if the service has not reached the given state within the deadline
   * @throws IllegalStateException if the service {@linkplain State#FAILED fails}.
   * @since 28.0
   */
