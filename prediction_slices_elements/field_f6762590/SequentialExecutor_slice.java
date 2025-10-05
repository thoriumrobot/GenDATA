// Source-based slice around line 87
// Method: com.google.common.util.concurrent.SequentialExecutor.worker

   * This counter prevents an ABA issue where a thread may successfully schedule the worker, the
   * worker runs and exhausts the queue, another thread enqueues a task and fails to schedule the
   * worker, and then the first thread's call to delegate.execute() returns. Without this counter,
   * it would observe the QUEUING state and set it to QUEUED, and the worker would never be
   * scheduled again for future submissions.
   */
  @GuardedBy("queue")
  private long workerRunCount = 0;

  @RetainedWith private final QueueWorker worker = new QueueWorker();

  /** Use {@link MoreExecutors#newSequentialExecutor} */
  SequentialExecutor(Executor executor) {
    this.executor = Preconditions.checkNotNull(executor);
  }

  /**
   * Adds a task to the queue and makes sure a worker thread is running.
   *
   * <p>If this method throws, e.g. a {@code RejectedExecutionException} from the delegate executor,
