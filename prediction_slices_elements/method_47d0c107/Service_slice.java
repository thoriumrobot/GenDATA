// Source-based slice around line 149
// Method: <com.google.common.util.concurrent.Service: void awaitTerminated(Duration)>

  /**
   * Waits for the {@link Service} to reach a terminal state (either {@link Service.State#TERMINATED
   * terminated} or {@link Service.State#FAILED failed}) for no more than the given time.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if the service has not reached the given state within the deadline
   * @throws IllegalStateException if the service {@linkplain State#FAILED fails}.
   * @since 28.0
   */
  default void awaitTerminated(Duration timeout) throws TimeoutException {
    awaitTerminated(toNanosSaturated(timeout), TimeUnit.NANOSECONDS);
  }

  /**
   * Waits for the {@link Service} to reach a terminal state (either {@link Service.State#TERMINATED
   * terminated} or {@link Service.State#FAILED failed}) for no more than the given time.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
   * @throws TimeoutException if the service has not reached the given state within the deadline
