// Source-based slice around line 67
// Method: com.google.common.util.concurrent.SequentialExecutor.executor

    IDLE,
    /** Runnable is not running, but is being queued for execution */
    QUEUING,
    /** runnable has been submitted but has not yet begun execution */
    QUEUED,
    RUNNING,
  }

  /** Underlying executor that all submitted Runnable objects are run on. */
  private final Executor executor;

  @GuardedBy("queue")
  private final Deque<Runnable> queue = new ArrayDeque<>();

  /** see {@link WorkerRunningState} */
  @LazyInit
  @GuardedBy("queue")
  private WorkerRunningState workerRunningState = IDLE;

  /**
