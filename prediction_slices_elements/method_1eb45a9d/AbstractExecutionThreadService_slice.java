// Source-based slice around line 97
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: void startUp()>


  /** Constructor for use by subclasses. */
  protected AbstractExecutionThreadService() {}

  /**
   * Start the service. This method is invoked on the execution thread.
   *
   * <p>By default this method does nothing.
   */
  protected void startUp() throws Exception {}

  /**
   * Run the service. This method is invoked on the execution thread. Implementations must respond
   * to stop requests. You could poll for lifecycle changes in a work loop:
   *
   * <pre>
   *   public void run() {
   *     while ({@link #isRunning()}) {
   *       // perform a unit of work
   *     }
