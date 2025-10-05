// Source-based slice around line 53
// Method: com.google.common.util.concurrent.ExecutionList.runnables

public final class ExecutionList {
  /** Logger to log exceptions caught when running runnables. */
  private static final LazyLogger log = new LazyLogger(ExecutionList.class);

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
