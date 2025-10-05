// Source-based slice around line 225
// Method: <com.google.common.util.concurrent.AbstractService: void doStop()>

   * <p>This method should return promptly; prefer to do work on a different thread where it is
   * convenient. It is invoked exactly once on service shutdown, even when {@link #stopAsync} is
   * called multiple times.
   *
   * <p>If {@link #stopAsync} is called on a {@link State#STARTING} service, this method is not
   * invoked immediately. Instead, it will be deferred until after the service is {@link
   * State#RUNNING}. Services that need to cancel startup work can override {@link #doCancelStart}.
   */
  @ForOverride
  protected abstract void doStop();

  /**
   * This method is called by {@link #stopAsync} when the service is still starting (i.e. {@link
   * #startAsync} has been called but {@link #notifyStarted} has not). Subclasses can override the
   * method to cancel pending work and then call {@link #notifyStopped} to stop the service.
   *
   * <p>This method should return promptly; prefer to do work on a different thread where it is
   * convenient. It is invoked exactly once on service shutdown, even when {@link #stopAsync} is
   * called multiple times.
   *
