// Source-based slice around line 438
// Method: <com.google.common.util.concurrent.ServiceManager: String toString()>

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

  /**
   * An encapsulation of all the mutable state of the {@link ServiceManager} that needs to be
   * accessed by instances of {@link ServiceListener}.
   */
  private static final class ServiceManagerState {
