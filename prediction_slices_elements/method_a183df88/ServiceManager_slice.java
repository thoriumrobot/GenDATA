// Source-based slice around line 330
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitHealthy(long,TimeUnit)>

   * reached the {@linkplain State#RUNNING running} state.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
   * @throws TimeoutException if not all of the services have finished starting within the deadline
   * @throws IllegalStateException if the service manager reaches a state from which it cannot
   *     become {@linkplain #isHealthy() healthy}.
   */
  @SuppressWarnings("GoodTime") // should accept a java.time.Duration
  public void awaitHealthy(long timeout, TimeUnit unit) throws TimeoutException {
    state.awaitHealthy(timeout, unit);
  }

  /**
   * Initiates service {@linkplain Service#stopAsync shutdown} if necessary on all the services
   * being managed.
   *
   * @return this
   */
  @CanIgnoreReturnValue
