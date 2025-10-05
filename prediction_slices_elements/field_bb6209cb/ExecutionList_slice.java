// Source-based slice around line 56
// Method: com.google.common.util.concurrent.ExecutionList.executed


  /**
   * The runnable, executor pairs to execute. This acts as a stack threaded through the {@link
   * RunnableExecutorPair#next} field.
   */
  @GuardedBy("this")
  private @Nullable RunnableExecutorPair runnables;

  @GuardedBy("this")
  private boolean executed;

  /** Creates a new, empty {@link ExecutionList}. */
  public ExecutionList() {}

  /**
   * Adds the {@code Runnable} and accompanying {@code Executor} to the list of listeners to
   * execute. If execution has already begun, the listener is executed immediately.
   *
   * <p>When selecting an executor, note that {@code directExecutor} is dangerous in some cases. See
   * the discussion in the {@link ListenableFuture#addListener ListenableFuture.addListener}
