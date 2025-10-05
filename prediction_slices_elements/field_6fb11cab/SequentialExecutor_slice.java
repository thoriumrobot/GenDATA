// Source-based slice around line 54
// Method: com.google.common.util.concurrent.SequentialExecutor.log

 * continues. See {@link QueueWorker#workOnQueue} for details.
 *
 * <p>{@code RuntimeException}s thrown by tasks are simply logged and the executor keeps trucking.
 * If an {@code Error} is thrown, the error will propagate and execution will stop until it is
 * restarted by a call to {@link #execute}.
 */
@J2ktIncompatible
@GwtIncompatible
final class SequentialExecutor implements Executor {
  private static final LazyLogger log = new LazyLogger(SequentialExecutor.class);

  enum WorkerRunningState {
    /** Runnable is not running and not queued for execution */
    IDLE,
    /** Runnable is not running, but is being queued for execution */
    QUEUING,
    /** runnable has been submitted but has not yet begun execution */
    QUEUED,
    RUNNING,
  }
