// Source-based slice around line 147
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: Executor executor()>

   * Returns the {@link Executor} that will be used to run this service. Subclasses may override
   * this method to use a custom {@link Executor}, which may configure its worker thread with a
   * specific name, thread group or priority. The returned executor's {@link
   * Executor#execute(Runnable) execute()} method is called when this service is started, and should
   * return promptly.
   *
   * <p>The default implementation returns a new {@link Executor} that sets the name of its threads
   * to the string returned by {@link #serviceName}
   */
  protected Executor executor() {
    return command -> newThread(serviceName(), command).start();
  }

  @Override
  public String toString() {
    return serviceName() + " [" + state() + "]";
  }

  @Override
  public final boolean isRunning() {
