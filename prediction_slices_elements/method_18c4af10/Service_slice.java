// Source-based slice around line 71
// Method: <com.google.common.util.concurrent.Service: Service startAsync()>

  /**
   * If the service state is {@link State#NEW}, this initiates service startup and returns
   * immediately. A stopped service may not be restarted.
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
