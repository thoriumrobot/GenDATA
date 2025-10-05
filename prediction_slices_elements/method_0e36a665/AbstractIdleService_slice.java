// Source-based slice around line 220
// Method: <com.google.common.util.concurrent.AbstractIdleService: String serviceName()>

    delegate.awaitTerminated(timeout, unit);
  }

  /**
   * Returns the name of this service. {@link AbstractIdleService} may include the name in debugging
   * output.
   *
   * @since 14.0
   */
  protected String serviceName() {
    return getClass().getSimpleName();
  }
}
