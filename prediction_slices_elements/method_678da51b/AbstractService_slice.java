// Source-based slice around line 242
// Method: <com.google.common.util.concurrent.AbstractService: void doCancelStart()>

   * convenient. It is invoked exactly once on service shutdown, even when {@link #stopAsync} is
   * called multiple times.
   *
   * <p>When this method is called {@link #state()} will return {@link State#STOPPING}, which is the
   * external state observable by the caller of {@link #stopAsync}.
   *
   * @since 27.0
   */
  @ForOverride
  protected void doCancelStart() {}

  @CanIgnoreReturnValue
  @Override
  public final Service startAsync() {
    if (monitor.enterIf(isStartable)) {
      try {
        snapshot = new StateSnapshot(STARTING);
        enqueueStartingEvent();
        doStart();
      } catch (Throwable startupFailure) {
