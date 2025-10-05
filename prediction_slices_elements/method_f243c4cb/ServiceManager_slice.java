// Source-based slice around line 432
// Method: <com.google.common.util.concurrent.ServiceManager: ImmutableMap startupDurations()>

  /**
   * Returns the service load times. This value will only return startup times for services that
   * have finished starting.
   *
   * @return Map of services and their corresponding startup time, the map entries will be ordered
   *     by startup time.
   * @since 31.0 (but only since 33.4.0 in the Android flavor)
   */
  @J2ObjCIncompatible
  public ImmutableMap<Service, Duration> startupDurations() {
    return ImmutableMap.copyOf(
        Maps.<Service, Long, Duration>transformValues(startupTimes(), Duration::ofMillis));
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(ServiceManager.class)
        .add("services", Collections2.filter(services, not(instanceOf(NoOpService.class))))
        .toString();
  }
