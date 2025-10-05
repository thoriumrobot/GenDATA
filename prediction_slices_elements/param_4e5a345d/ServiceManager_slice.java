// Source-based slice around line 380
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitStopped(long,TimeUnit)>

   * Waits for the all the services to reach a terminal state for no more than the given time. After
   * this method returns all services will either be {@linkplain Service.State#TERMINATED
   * terminated} or {@linkplain Service.State#FAILED failed}.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
   * @throws TimeoutException if not all of the services have stopped within the deadline
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public void awaitStopped(long timeout, TimeUnit unit) throws TimeoutException {
    state.awaitStopped(timeout, unit);
  }

  /**
   * Returns true if all services are currently in the {@linkplain State#RUNNING running} state.
   *
   * <p>Users who want more detailed information should use the {@link #servicesByState} method to
   * get detailed information about which services are not running.
   */
  public boolean isHealthy() {
