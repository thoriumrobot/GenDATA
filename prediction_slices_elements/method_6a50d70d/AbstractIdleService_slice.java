// Source-based slice around line 116
// Method: <com.google.common.util.concurrent.AbstractIdleService: String toString()>

   * specific name, thread group or priority. The returned executor's {@link
   * Executor#execute(Runnable) execute()} method is called when this service is started and
   * stopped, and should return promptly.
   */
  protected Executor executor() {
    return command -> newThread(threadNameSupplier.get(), command).start();
  }

  @Override
  public String toString() {
    return serviceName() + " [" + state() + "]";
  }

  @Override
  public final boolean isRunning() {
    return delegate.isRunning();
  }

  @Override
  public final State state() {
