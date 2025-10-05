// Source-based slice around line 193
// Method: <com.google.common.base.Stopwatch: Stopwatch reset()>

    return this;
  }

  /**
   * Sets the elapsed time for this stopwatch to zero, and places it in a stopped state.
   *
   * @return this {@code Stopwatch} instance
   */
  @CanIgnoreReturnValue
  public Stopwatch reset() {
    elapsedNanos = 0;
    isRunning = false;
    return this;
  }

  private long elapsedNanos() {
    return isRunning ? ticker.read() - startTick + elapsedNanos : elapsedNanos;
  }

  /**
