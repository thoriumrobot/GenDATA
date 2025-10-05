// Source-based slice around line 246
// Method: <com.google.common.util.concurrent.AbstractService: Service startAsync()>

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
        restoreInterruptIfIsInterruptedException(startupFailure);
        notifyFailed(startupFailure);
      } finally {
        monitor.leave();
