// Source-based slice around line 299
// Method: <com.google.common.util.concurrent.ServiceManager: void awaitHealthy()>


  /**
   * Waits for the {@link ServiceManager} to become {@linkplain #isHealthy() healthy}. The manager
   * will become healthy after all the component services have reached the {@linkplain State#RUNNING
   * running} state.
   *
   * @throws IllegalStateException if the service manager reaches a state from which it cannot
   *     become {@linkplain #isHealthy() healthy}.
   */
  public void awaitHealthy() {
    state.awaitHealthy();
  }

  /**
   * Waits for the {@link ServiceManager} to become {@linkplain #isHealthy() healthy} for no more
   * than the given time. The manager will become healthy after all the component services have
   * reached the {@linkplain State#RUNNING running} state.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if not all of the services have finished starting within the deadline
