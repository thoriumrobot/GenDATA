// Source-based slice around line 114
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void run()>

   *     while ({@link #isRunning()}) {
   *       // perform a unit of work
   *     }
   *   }
   * </pre>
   *
   * <p>...or you could respond to stop requests by implementing {@link #triggerShutdown()}, which
   * should cause {@link #run()} to return.
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
