// Source-based slice around line 463
// Method: <com.google.common.util.concurrent.AbstractService: void notifyFailed(Throwable)>

      dispatchListenerEvents();
    }
  }

  /**
   * Invoke this method to transition the service to the {@link State#FAILED}. The service will
   * <b>not be stopped</b> if it is running. Invoke this method when a service has failed critically
   * or otherwise cannot be started nor stopped.
   */
  protected final void notifyFailed(Throwable cause) {
    checkNotNull(cause);

    monitor.enter();
    try {
      State previous = state();
      switch (previous) {
        case NEW:
        case TERMINATED:
          throw new IllegalStateException("Failed while in state:" + previous, cause);
        case RUNNING:
