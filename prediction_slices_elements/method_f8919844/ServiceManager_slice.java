// Source-based slice around line 272
// Method: <com.google.common.util.concurrent.ServiceManager: ServiceManager startAsync()>

  /**
   * Initiates service {@linkplain Service#startAsync startup} on all the services being managed. It
   * is only valid to call this method if all of the services are {@linkplain State#NEW new}.
   *
   * @return this
   * @throws IllegalStateException if any of the Services are not {@link State#NEW new} when the
   *     method is called.
   */
  @CanIgnoreReturnValue
  public ServiceManager startAsync() {
    for (Service service : services) {
      checkState(service.state() == NEW, "Not all services are NEW, cannot start %s", this);
    }
    for (Service service : services) {
      try {
        state.tryStartTiming(service);
        service.startAsync();
      } catch (IllegalStateException e) {
        // This can happen if the service has already been started or stopped (e.g. by another
        // service or listener). Our contract says it is safe to call this method if
