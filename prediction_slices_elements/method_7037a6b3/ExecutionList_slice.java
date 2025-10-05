// Source-based slice around line 102
// Method: <com.google.common.util.concurrent.ExecutionList: void execute()>

   * note that listeners added after this point may be executed before those previously added, and
   * note that the execution order of all listeners is ultimately chosen by the implementations of
   * the supplied executors.
   *
   * <p>This method is idempotent. Calling it several times in parallel is semantically equivalent
   * to calling it exactly once.
   *
   * @since 10.0 (present in 1.0 as {@code run})
   */
  public void execute() {
    // Lock while we update our state so the add method above will finish adding any listeners
    // before we start to run them.
    RunnableExecutorPair list;
    synchronized (this) {
      if (executed) {
        return;
      }
      executed = true;
      list = runnables;
      runnables = null; // allow GC to free listeners even if this stays around for a while.
