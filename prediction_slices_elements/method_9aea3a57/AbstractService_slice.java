// Source-based slice around line 436
// Method: <com.google.common.util.concurrent.AbstractService: void notifyStopped()>


  /**
   * Implementing classes should invoke this method once their service has stopped. It will cause
   * the service to transition from {@link State#STARTING} or {@link State#STOPPING} to {@link
   * State#TERMINATED}.
   *
   * @throws IllegalStateException if the service is not one of {@link State#STOPPING}, {@link
   *     State#STARTING}, or {@link State#RUNNING}.
   */
  protected final void notifyStopped() {
    monitor.enter();
    try {
      State previous = state();
      switch (previous) {
        case NEW:
        case TERMINATED:
        case FAILED:
          throw new IllegalStateException("Cannot notifyStopped() when the service is " + previous);
        case RUNNING:
        case STARTING:
