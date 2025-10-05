// Source-based slice around line 419
// Method: <com.google.common.util.concurrent.ServiceManager: ImmutableMap startupTimes()>

  }

  /**
   * Returns the service load times. This value will only return startup times for services that
   * have finished starting.
   *
   * @return Map of services and their corresponding startup time in millis, the map entries will be
   *     ordered by startup time.
   */
  public ImmutableMap<Service, Long> startupTimes() {
    return state.startupTimes();
  }

  /**
   * Returns the service load times. This value will only return startup times for services that
   * have finished starting.
   *
   * @return Map of services and their corresponding startup time, the map entries will be ordered
   *     by startup time.
   * @since 31.0 (but only since 33.4.0 in the Android flavor)
