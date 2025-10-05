// Source-based slice around line 408
// Method: <com.google.common.util.concurrent.ServiceManager: ImmutableSetMultimap servicesByState()>

  /**
   * Provides a snapshot of the current state of all the services under management.
   *
   * <p>N.B. This snapshot is guaranteed to be consistent, i.e. the set of states returned will
   * correspond to a point in time view of the services.
   *
   * @since 29.0 (present with return type {@code ImmutableMultimap} since 14.0)
   */
  @Override
  public ImmutableSetMultimap<State, Service> servicesByState() {
    return state.servicesByState();
  }

  /**
   * Returns the service load times. This value will only return startup times for services that
   * have finished starting.
   *
   * @return Map of services and their corresponding startup time in millis, the map entries will be
   *     ordered by startup time.
   */
