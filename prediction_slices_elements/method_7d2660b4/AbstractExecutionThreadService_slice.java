// Source-based slice around line 122
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void shutDown()>

   */
  protected abstract void run() throws Exception;

  /**
   * Stop the service. This method is invoked on the execution thread.
   *
   * <p>By default this method does nothing.
   */
  // TODO: consider supporting a TearDownTestCase-like API
  protected void shutDown() throws Exception {}

  /**
   * Invoked to request the service to stop.
   *
   * <p>By default this method does nothing.
   *
   * <p>Currently, this method is invoked while holding a lock. If an implementation of this method
   * blocks, it can prevent this service from changing state. If you need to performing a blocking
   * operation in order to trigger shutdown, consider instead registering a listener and
   * implementing {@code stopping}. Note, however, that {@code stopping} does not run at exactly the
