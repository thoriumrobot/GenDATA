// Source-based slice around line 69
// Method: <com.google.common.util.concurrent.ExecutionList: void add(Runnable,Executor)>


  /**
   * Adds the {@code Runnable} and accompanying {@code Executor} to the list of listeners to
   * execute. If execution has already begun, the listener is executed immediately.
   *
   * <p>When selecting an executor, note that {@code directExecutor} is dangerous in some cases. See
   * the discussion in the {@link ListenableFuture#addListener ListenableFuture.addListener}
   * documentation.
   */
  public void add(Runnable runnable, Executor executor) {
    // Fail fast on a null. We throw NPE here because the contract of Executor states that it throws
    // NPE on null listener, so we propagate that contract up into the add method as well.
    checkNotNull(runnable, "Runnable was null.");
    checkNotNull(executor, "Executor was null.");

    // Lock while we check state. We must maintain the lock while adding the new pair so that
    // another thread can't run the list out from under us. We only add to the list if we have not
    // yet started execution.
    synchronized (this) {
      if (!executed) {
