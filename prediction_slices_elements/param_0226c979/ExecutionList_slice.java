// Source-based slice around line 141
// Method: <com.google.common.util.concurrent.ExecutionList: void executeListener(Runnable,Executor)>

      reversedList = reversedList.next;
    }
  }

  /**
   * Submits the given runnable to the given {@link Executor} catching and logging all {@linkplain
   * RuntimeException runtime exceptions} thrown by the executor.
   */
  @SuppressWarnings("CatchingUnchecked") // sneaky checked exception
  private static void executeListener(Runnable runnable, Executor executor) {
    try {
      executor.execute(runnable);
    } catch (Exception e) { // sneaky checked exception
      // Log it and keep going -- bad runnable and/or executor. Don't punish the other runnables if
      // we're given a bad one. We only catch Exception because we want Errors to propagate up.
      log.get()
          .log(
              Level.SEVERE,
              "RuntimeException while executing runnable "
                  + runnable
