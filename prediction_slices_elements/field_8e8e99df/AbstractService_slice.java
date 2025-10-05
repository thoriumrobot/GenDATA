// Source-based slice around line 192
// Method: com.google.common.util.concurrent.AbstractService.snapshot

  /**
   * The current state of the service. This should be written with the lock held but can be read
   * without it because it is an immutable object in a volatile field. This is desirable so that
   * methods like {@link #state}, {@link #failureCause} and notably {@link #toString} can be run
   * without grabbing the lock.
   *
   * <p>To update this field correctly the lock must be held to guarantee that the state is
   * consistent.
   */
  private volatile StateSnapshot snapshot = new StateSnapshot(NEW);

  /** Constructor for use by subclasses. */
  protected AbstractService() {}

  /**
   * This method is called by {@link #startAsync} to initiate service startup. The invocation of
   * this method should cause a call to {@link #notifyStarted()}, either during this method's run,
   * or after it has returned. If startup fails, the invocation should cause a call to {@link
   * #notifyFailed(Throwable)} instead.
   *
