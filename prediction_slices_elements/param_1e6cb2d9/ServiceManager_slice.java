// Source-based slice around line 314
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitHealthy(Duration)>

   * than the given time. The manager will become healthy after all the component services have
   * reached the {@linkplain State#RUNNING running} state.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if not all of the services have finished starting within the deadline
   * @throws IllegalStateException if the service manager reaches a state from which it cannot
   *     become {@linkplain #isHealthy() healthy}.
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public void awaitHealthy(Duration timeout) throws TimeoutException {
    awaitHealthy(toNanosSaturated(timeout), TimeUnit.NANOSECONDS);
  }

  /**
   * Waits for the {@link ServiceManager} to become {@linkplain #isHealthy() healthy} for no more
   * than the given time. The manager will become healthy after all the component services have
   * reached the {@linkplain State#RUNNING running} state.
   *
   * @param timeout the maximum time to wait
   * @param unit the time unit of the timeout argument
