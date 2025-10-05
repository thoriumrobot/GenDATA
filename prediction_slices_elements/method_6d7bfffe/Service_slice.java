// Source-based slice around line 100
// Method: <com.google.common.util.concurrent.Service: void awaitRunning()>


  /**
   * Waits for the {@link Service} to reach the {@linkplain State#RUNNING running state}.
   *
   * @throws IllegalStateException if the service reaches a state from which it is not possible to
   *     enter the {@link State#RUNNING} state. e.g. if the {@code state} is {@code
   *     State#TERMINATED} when this method is called then this will throw an IllegalStateException.
   * @since 15.0
   */
  void awaitRunning();

  /**
   * Waits for the {@link Service} to reach the {@linkplain State#RUNNING running state} for no more
   * than the given time.
   *
   * @param timeout the maximum time to wait
   * @throws TimeoutException if the service has not reached the given state within the deadline
   * @throws IllegalStateException if the service reaches a state from which it is not possible to
   *     enter the {@link State#RUNNING RUNNING} state. e.g. if the {@code state} is {@code
   *     State#TERMINATED} when this method is called then this will throw an IllegalStateException.
