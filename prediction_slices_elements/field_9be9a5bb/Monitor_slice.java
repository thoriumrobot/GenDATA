// Source-based slice around line 332
// Method: com.google.common.util.concurrent.Monitor.fair

    /**
     * Evaluates this guard's boolean condition. This method is always called with the associated
     * monitor already occupied. Implementations of this method must depend only on state protected
     * by the associated monitor, and must not modify that state.
     */
    public abstract boolean isSatisfied();
  }

  /** Whether this monitor is fair. */
  private final boolean fair;

  /** The lock underlying this monitor. */
  private final ReentrantLock lock;

  /**
   * The guards associated with this monitor that currently have waiters ({@code waiterCount > 0}).
   * A linked list threaded through the Guard.next field.
   */
  @GuardedBy("lock")
  private @Nullable Guard activeGuards = null;
