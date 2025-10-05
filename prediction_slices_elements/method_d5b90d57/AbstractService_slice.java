// Source-based slice around line 267
// Method: <com.google.common.util.concurrent.AbstractService: Service stopAsync()>

      }
    } else {
      throw new IllegalStateException("Service " + this + " has already been started");
    }
    return this;
  }

  @CanIgnoreReturnValue
  @Override
  public final Service stopAsync() {
    if (monitor.enterIf(isStoppable)) {
      try {
        State previous = state();
        switch (previous) {
          case NEW:
            snapshot = new StateSnapshot(TERMINATED);
            enqueueTerminatedEvent(NEW);
            break;
          case STARTING:
            snapshot = new StateSnapshot(STARTING, true, null);
