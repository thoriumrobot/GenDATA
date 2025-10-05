// Source-based slice around line 101
// Method: <com.google.common.util.concurrent.SequentialExecutor: void execute(Runnable)>

  }

  /**
   * Adds a task to the queue and makes sure a worker thread is running.
   *
   * <p>If this method throws, e.g. a {@code RejectedExecutionException} from the delegate executor,
   * execution of tasks will stop until a call to this method is made.
   */
  @Override
  public void execute(Runnable task) {
    checkNotNull(task);
    Runnable submittedTask;
    long oldRunCount;
    synchronized (queue) {
      // If the worker is already running (or execute() on the delegate returned successfully, and
      // the worker has yet to start) then we don't need to start the worker.
      if (workerRunningState == RUNNING || workerRunningState == QUEUED) {
        queue.add(task);
        return;
      }
