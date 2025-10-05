// Source-based slice around line 49
// Method: <com.google.common.util.concurrent.DirectExecutorService: void execute(Runnable)>

   *   - Terminated: runningTasks == 0 and shutdown == true
   */
  @GuardedBy("lock")
  private int runningTasks = 0;

  @GuardedBy("lock")
  private boolean shutdown = false;

  @Override
  public void execute(Runnable command) {
    startTask();
    try {
      command.run();
    } finally {
      endTask();
    }
  }

  @Override
  public boolean isShutdown() {
