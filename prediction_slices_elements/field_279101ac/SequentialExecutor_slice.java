// Source-based slice around line 75
// Method: com.google.common.util.concurrent.SequentialExecutor.workerRunningState

  /** Underlying executor that all submitted Runnable objects are run on. */
  private final Executor executor;

  @GuardedBy("queue")
  private final Deque<Runnable> queue = new ArrayDeque<>();

  /** see {@link WorkerRunningState} */
  @LazyInit
  @GuardedBy("queue")
  private WorkerRunningState workerRunningState = IDLE;

  /**
   * This counter prevents an ABA issue where a thread may successfully schedule the worker, the
   * worker runs and exhausts the queue, another thread enqueues a task and fails to schedule the
   * worker, and then the first thread's call to delegate.execute() returns. Without this counter,
   * it would observe the QUEUING state and set it to QUEUED, and the worker would never be
   * scheduled again for future submissions.
   */
  @GuardedBy("queue")
  private long workerRunCount = 0;
