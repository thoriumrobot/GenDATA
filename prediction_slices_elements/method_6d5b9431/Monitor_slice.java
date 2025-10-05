// Source-based slice around line 381
// Method: <com.google.common.util.concurrent.Monitor: void enter()>

    return new Guard(this) {
      @Override
      public boolean isSatisfied() {
        return isSatisfied.getAsBoolean();
      }
    };
  }

  /** Enters this monitor. Blocks indefinitely. */
  public void enter() {
    lock.lock();
  }

  /**
   * Enters this monitor. Blocks at most the given time.
   *
   * @return whether the monitor was entered
   * @since 28.0 (but only since 33.4.0 in the Android flavor)
   */
  public boolean enter(Duration time) {
