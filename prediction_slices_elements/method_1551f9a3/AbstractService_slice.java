// Source-based slice around line 495
// Method: <com.google.common.util.concurrent.AbstractService: State state()>

    }
  }

  @Override
  public final boolean isRunning() {
    return state() == RUNNING;
  }

  @Override
  public final State state() {
    return snapshot.externalState();
  }

  /**
   * @since 14.0
   */
  @Override
  public final Throwable failureCause() {
    return snapshot.failureCause();
  }
