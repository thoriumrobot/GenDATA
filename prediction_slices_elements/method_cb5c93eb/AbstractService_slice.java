// Source-based slice around line 380
// Method: <com.google.common.util.concurrent.AbstractService: void checkCurrentState(State)>

              + this
              + " to reach a terminal state. "
              + "Current state: "
              + state());
    }
  }

  /** Checks that the current state is equal to the expected state. */
  @GuardedBy("monitor")
  private void checkCurrentState(State expected) {
    State actual = state();
    if (actual != expected) {
      if (actual == FAILED) {
        // Handle this specially so that we can include the failureCause, if there is one.
        throw new IllegalStateException(
            "Expected the service " + this + " to be " + expected + ", but the service has FAILED",
            failureCause());
      }
      throw new IllegalStateException(
          "Expected the service " + this + " to be " + expected + ", but was " + actual);
