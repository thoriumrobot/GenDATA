// Source-based slice around line 74
// Method: <com.google.common.util.concurrent.Service: boolean isRunning()>

   *
   * @return this
   * @throws IllegalStateException if the service is not {@link State#NEW}
   * @since 15.0
   */
  @CanIgnoreReturnValue
  Service startAsync();

  /** Returns {@code true} if this service is {@linkplain State#RUNNING running}. */
  boolean isRunning();

  /** Returns the lifecycle state of the service. */
  State state();

  /**
   * If the service is {@linkplain State#STARTING starting} or {@linkplain State#RUNNING running},
   * this initiates service shutdown and returns immediately. If the service is {@linkplain
   * State#NEW new}, it is {@linkplain State#TERMINATED terminated} without having been started nor
   * stopped. If the service has already been stopped, this method returns immediately without
   * taking action.
