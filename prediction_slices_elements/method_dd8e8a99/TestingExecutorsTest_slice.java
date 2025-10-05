// Source-based slice around line 83
// Method: <com.google.common.util.concurrent.testing.TestingExecutorsTest: void testSameThreadScheduledExecutor()>

          }
        };
    List<Future<Boolean>> futureList = executor.invokeAll(ImmutableList.of(task), 10, MILLISECONDS);
    Future<Boolean> future = futureList.get(0);
    assertFalse(taskDone);
    assertTrue(future.isDone());
    assertThrows(CancellationException.class, () -> future.get());
  }

  public void testSameThreadScheduledExecutor() throws ExecutionException, InterruptedException {
    taskDone = false;
    Callable<Integer> task =
        new Callable<Integer>() {
          @Override
          public Integer call() {
            taskDone = true;
            return 6;
          }
        };
    Future<Integer> future =
