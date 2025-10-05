// Source-based slice around line 152
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: String toString()>

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
    return delegate.isRunning();
  }

  @Override
  public final State state() {
