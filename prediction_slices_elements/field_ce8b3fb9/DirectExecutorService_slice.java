// Source-based slice around line 33
// Method: com.google.common.util.concurrent.DirectExecutorService.lock

import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

/** See newDirectExecutorService javadoc for behavioral notes. */
@J2ktIncompatible // Emulated
@GwtIncompatible
final class DirectExecutorService extends AbstractListeningExecutorService {

  /** Lock used whenever accessing the state variables (runningTasks, shutdown) of the executor */
  private final Object lock = new Object();

  /*
   * Conceptually, these two variables describe the executor being in
   * one of three states:
   *   - Active: shutdown == false
   *   - Shutdown: runningTasks > 0 and shutdown == true
   *   - Terminated: runningTasks == 0 and shutdown == true
   */
  @GuardedBy("lock")
  private int runningTasks = 0;
