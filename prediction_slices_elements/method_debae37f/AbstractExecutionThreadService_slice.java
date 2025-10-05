// Source-based slice around line 258
// Method: <com.google.common.util.concurrent.AbstractExecutionThreadService: String serviceName()>


  /**
   * Returns the name of this service. {@link AbstractExecutionThreadService} may include the name
   * in debugging output.
   *
   * <p>Subclasses may override this method.
   *
   * @since 14.0 (present in 10.0 as getServiceName)
   */
  protected String serviceName() {
    return getClass().getSimpleName();
  }
}
