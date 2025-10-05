// Source-based slice around line 65
// Method: <com.google.common.util.concurrent.testing.TestingExecutorsTest: void testNoOpScheduledExecutorInvokeAll()>

  public void testNoOpScheduledExecutorShutdown() {
    ListeningScheduledExecutorService executor = TestingExecutors.noOpScheduledExecutor();
    assertFalse(executor.isShutdown());
    assertFalse(executor.isTerminated());
    executor.shutdown();
    assertTrue(executor.isShutdown());
    assertTrue(executor.isTerminated());
  }

  public void testNoOpScheduledExecutorInvokeAll() throws ExecutionException, InterruptedException {
    ListeningScheduledExecutorService executor = TestingExecutors.noOpScheduledExecutor();
    taskDone = false;
    Callable<Boolean> task =
        new Callable<Boolean>() {
          @Override
          public Boolean call() {
            taskDone = true;
            return taskDone;
          }
        };
