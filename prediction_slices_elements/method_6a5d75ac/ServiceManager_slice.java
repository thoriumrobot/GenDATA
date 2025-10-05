// Source-based slice around line 259
// Method: <com.google.common.util.concurrent.ServiceManager: void addListener(Listener,Executor)>

   * logged.
   *
   * <p>When selecting an executor, note that {@code directExecutor} is dangerous in some cases. See
   * the discussion in the {@link ListenableFuture#addListener ListenableFuture.addListener}
   * documentation.
   *
   * @param listener the listener to run when the manager changes state
   * @param executor the executor in which the listeners callback methods will be run.
   */
  public void addListener(Listener listener, Executor executor) {
    state.addListener(listener, executor);
  }

  /**
   * Initiates service {@linkplain Service#startAsync startup} on all the services being managed. It
   * is only valid to call this method if all of the services are {@linkplain State#NEW new}.
   *
   * @return this
   * @throws IllegalStateException if any of the Services are not {@link State#NEW new} when the
   *     method is called.
