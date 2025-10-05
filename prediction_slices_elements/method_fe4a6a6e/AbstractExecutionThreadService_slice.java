// Source-based slice around line 135
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void triggerShutdown()>

   *
   * <p>By default this method does nothing.
   *
   * <p>Currently, this method is invoked while holding a lock. If an implementation of this method
   * blocks, it can prevent this service from changing state. If you need to performing a blocking
   * operation in order to trigger shutdown, consider instead registering a listener and
   * implementing {@code stopping}. Note, however, that {@code stopping} does not run at exactly the
   * same times as {@code triggerShutdown}.
   */
  protected void triggerShutdown() {}

  /**
   * Returns the {@link Executor} that will be used to run this service. Subclasses may override
   * this method to use a custom {@link Executor}, which may configure its worker thread with a
   * specific name, thread group or priority. The returned executor's {@link
   * Executor#execute(Runnable) execute()} method is called when this service is started, and should
   * return promptly.
   *
   * <p>The default implementation returns a new {@link Executor} that sets the name of its threads
   * to the string returned by {@link #serviceName}
