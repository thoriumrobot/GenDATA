// Source-based slice around line 197
// Method: <com.google.common.util.concurrent.Service: void addListener(Listener,Executor)>

   * during {@code Executor.execute} (e.g., a {@code RejectedExecutionException}) will be caught and
   * logged.
   *
   * @param listener the listener to run when the service changes state is complete
   * @param executor the executor in which the listeners callback methods will be run. For fast,
   *     lightweight listeners that would be safe to execute in any thread, consider {@link
   *     MoreExecutors#directExecutor}.
   * @since 13.0
   */
  void addListener(Listener listener, Executor executor);

  /**
   * The lifecycle states of a service.
   *
   * <p>The ordering of the {@link State} enum is defined such that if there is a state transition
   * from {@code A -> B} then {@code A.compareTo(B) < 0}. N.B. The converse is not true, i.e. if
   * {@code A.compareTo(B) < 0} then there is <b>not</b> guaranteed to be a valid state transition
   * {@code A -> B}.
   *
   * @since 9.0 (in 1.0 as {@code com.google.common.base.Service.State})
