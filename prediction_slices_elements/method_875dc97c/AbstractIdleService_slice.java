// Source-based slice around line 102
// Method: <com.google.common.util.concurrent.AbstractIdleService: void shutDown()>

  }

  /** Constructor for use by subclasses. */
  protected AbstractIdleService() {}

  /** Start the service. */
  protected abstract void startUp() throws Exception;

  /** Stop the service. */
  protected abstract void shutDown() throws Exception;

  /**
   * Returns the {@link Executor} that will be used to run this service. Subclasses may override
   * this method to use a custom {@link Executor}, which may configure its worker thread with a
   * specific name, thread group or priority. The returned executor's {@link
   * Executor#execute(Runnable) execute()} method is called when this service is started and
   * stopped, and should return promptly.
   */
  protected Executor executor() {
    return command -> newThread(threadNameSupplier.get(), command).start();
