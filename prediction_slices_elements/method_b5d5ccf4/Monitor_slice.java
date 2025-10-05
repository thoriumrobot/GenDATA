// Source-based slice around line 370
// Method: <com.google.common.util.concurrent.Monitor: Guard newGuard(BooleanSupplier)>

  }

  /**
   * Creates a new {@linkplain Guard guard} for this monitor.
   *
   * @param isSatisfied the new guard's boolean condition (see {@link Guard#isSatisfied
   *     isSatisfied()})
   * @since 21.0 (but only since 33.4.0 in the Android flavor)
   */
  public Guard newGuard(BooleanSupplier isSatisfied) {
    checkNotNull(isSatisfied, "isSatisfied");
    return new Guard(this) {
      @Override
      public boolean isSatisfied() {
        return isSatisfied.getAsBoolean();
      }
    };
  }

  /** Enters this monitor. Blocks indefinitely. */
