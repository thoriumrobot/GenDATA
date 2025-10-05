// Source-based slice around line 99
// Method: <com.google.common.util.concurrent.testing.TestingExecutorsTest: void testSameThreadScheduledExecutorWithException()>

            return 6;
          }
        };
    Future<Integer> future =
        TestingExecutors.sameThreadScheduledExecutor().schedule(task, 10000, MILLISECONDS);
    assertTrue("Should run callable immediately", taskDone);
    assertEquals(6, (int) future.get());
  }

  public void testSameThreadScheduledExecutorWithException() throws InterruptedException {
    Runnable runnable =
        new Runnable() {
          @Override
          public void run() {
            throw new RuntimeException("Oh no!");
          }
        };

    Future<?> future = TestingExecutors.sameThreadScheduledExecutor().submit(runnable);
    assertThrows(ExecutionException.class, () -> future.get());
