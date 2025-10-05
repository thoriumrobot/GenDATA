// Source-based slice around line 90
// Method: <com.google.common.util.concurrent.Service: Service stopAsync()>

   * this initiates service shutdown and returns immediately. If the service is {@linkplain
   * State#NEW new}, it is {@linkplain State#TERMINATED terminated} without having been started nor
   * stopped. If the service has already been stopped, this method returns immediately without
   * taking action.
   *
   * @return this
   * @since 15.0
   */
  @CanIgnoreReturnValue
  Service stopAsync();

  /**
   * Waits for the {@link Service} to reach the {@linkplain State#RUNNING running state}.
   *
   * @throws IllegalStateException if the service reaches a state from which it is not possible to
   *     enter the {@link State#RUNNING} state. e.g. if the {@code state} is {@code
   *     State#TERMINATED} when this method is called then this will throw an IllegalStateException.
   * @since 15.0
   */
  void awaitRunning();
