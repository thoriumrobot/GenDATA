// Source-based slice around line 164
// Method: <com.google.common.base.Stopwatch: Stopwatch start()>

  }

  /**
   * Starts the stopwatch.
   *
   * @return this {@code Stopwatch} instance
   * @throws IllegalStateException if the stopwatch is already running.
   */
  @CanIgnoreReturnValue
  public Stopwatch start() {
    checkState(!isRunning, "This stopwatch is already running.");
    isRunning = true;
    startTick = ticker.read();
    return this;
  }

  /**
   * Stops the stopwatch. Future reads will return the fixed duration that had elapsed up to this
   * point.
   *
