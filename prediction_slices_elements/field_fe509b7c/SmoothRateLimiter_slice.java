// Source-based slice around line 315
// Method: com.google.common.util.concurrent.SmoothRateLimiter.storedPermits

    }

    @Override
    double coolDownIntervalMicros() {
      return stableIntervalMicros;
    }
  }

  /** The currently stored permits. */
  double storedPermits;

  /** The maximum number of stored permits. */
  double maxPermits;

  /**
   * The interval between two unit requests, at our stable rate. E.g., a stable rate of 5 permits
   * per second has a stable interval of 200ms.
   */
  double stableIntervalMicros;

