// Source-based slice around line 46
// Method: com.google.common.util.concurrent.ExecutionList.log

 *
 * @author Nishant Thakkar
 * @author Sven Mawson
 * @since 1.0
 */
@J2ktIncompatible
@GwtIncompatible
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
