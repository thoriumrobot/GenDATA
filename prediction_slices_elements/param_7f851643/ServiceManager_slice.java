// Source-based slice around line 366
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitStopped(Duration)>

  /**
   * Waits for the all the services to reach a terminal state for no more than the given time. After
   * this method returns all services will either be {@linkplain Service.State#TERMINATED
   * terminated} or {@linkplain Service.State#FAILED failed}.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if not all of the services have stopped within the deadline
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public void awaitStopped(Duration timeout) throws TimeoutException {
    awaitStopped(toNanosSaturated(timeout), TimeUnit.NANOSECONDS);
  }

  /**
   * Waits for the all the services to reach a terminal state for no more than the given time. After
   * this method returns all services will either be {@linkplain Service.State#TERMINATED
   * terminated} or {@linkplain Service.State#FAILED failed}.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
