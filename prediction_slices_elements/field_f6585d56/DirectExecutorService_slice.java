// Source-based slice around line 43
// Method: com.google.common.util.concurrent.DirectExecutorService.runningTasks


  /*
   * Conceptually, these two variables describe the executor being in
   * one of three states:
   *   - Active: shutdown == false
   *   - Shutdown: runningTasks > 0 and shutdown == true
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
