// Source-based slice around line 56
// Method: <com.google.common.util.concurrent.testing.TestingExecutorsTest: void testNoOpScheduledExecutorShutdown()>

          }
        };
    ScheduledFuture<?> future =
        TestingExecutors.noOpScheduledExecutor().schedule(task, 10, MILLISECONDS);
    Thread.sleep(20);
    assertFalse(taskDone);
    assertFalse(future.isDone());
  }

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
