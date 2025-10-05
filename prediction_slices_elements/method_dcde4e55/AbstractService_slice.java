// Source-based slice around line 400
// Method: <com.google.common.util.concurrent.AbstractService: void notifyStarted()>

    }
  }

  /**
   * Implementing classes should invoke this method once their service has started. It will cause
   * the service to transition from {@link State#STARTING} to {@link State#RUNNING}.
   *
   * @throws IllegalStateException if the service is not {@link State#STARTING}.
   */
  protected final void notifyStarted() {
    monitor.enter();
    try {
      // We have to examine the internal state of the snapshot here to properly handle the stop
      // while starting case.
      if (snapshot.state != STARTING) {
        IllegalStateException failure =
            new IllegalStateException(
                "Cannot notifyStarted() when the service is " + snapshot.state);
        notifyFailed(failure);
        throw failure;
