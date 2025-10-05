// Source-based slice around line 341
// Method: <com.google.common.util.concurrent.ServiceManager: ServiceManager stopAsync()>

  }

  /**
   * Initiates service {@linkplain Service#stopAsync shutdown} if necessary on all the services
   * being managed.
   *
   * @return this
   */
  @CanIgnoreReturnValue
  public ServiceManager stopAsync() {
    for (Service service : services) {
      service.stopAsync();
    }
    return this;
  }

  /**
   * Waits for the all the services to reach a terminal state. After this method returns all
   * services will either be {@linkplain Service.State#TERMINATED terminated} or {@linkplain
   * Service.State#FAILED failed}.
