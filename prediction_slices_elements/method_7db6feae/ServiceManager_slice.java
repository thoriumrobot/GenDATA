// Source-based slice around line 390
// Method: <com.google.common.util.concurrent.ServiceManager: boolean isHealthy()>

    state.awaitStopped(timeout, unit);
  }

  /**
   * Returns true if all services are currently in the {@linkplain State#RUNNING running} state.
   *
   * <p>Users who want more detailed information should use the {@link #servicesByState} method to
   * get detailed information about which services are not running.
   */
  public boolean isHealthy() {
    for (Service service : services) {
      if (!service.isRunning()) {
        return false;
      }
    }
    return true;
  }

  /**
   * Provides a snapshot of the current state of all the services under management.
