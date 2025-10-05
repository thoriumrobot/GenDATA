// Source-based slice around line 130
// Method: <com.google.common.util.concurrent.Service: void awaitRunning(long,TimeUnit)>

   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
   * @throws TimeoutException if the service has not reached the given state within the deadline
   * @throws IllegalStateException if the service reaches a state from which it is not possible to
   *     enter the {@link State#RUNNING RUNNING} state. e.g. if the {@code state} is {@code
   *     State#TERMINATED} when this method is called then this will throw an IllegalStateException.
   * @since 15.0
   */
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
