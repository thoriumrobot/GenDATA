// Source-based slice around line 342
// Method: com.google.common.util.concurrent.Monitor.activeGuards


  /** The lock underlying this monitor. */
  private final ReentrantLock lock;

  /**
   * The guards associated with this monitor that currently have waiters ({@code waiterCount > 0}).
   * A linked list threaded through the Guard.next field.
   */
  @GuardedBy("lock")
  private @Nullable Guard activeGuards = null;

  /**
   * Creates a monitor with a non-fair (but fast) ordering policy. Equivalent to {@code
   * Monitor(false)}.
   */
  public Monitor() {
    this(false);
  }

  /**
