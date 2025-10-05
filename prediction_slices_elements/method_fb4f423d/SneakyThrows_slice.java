// Source-based slice around line 46
// Method: <com.google.common.base.SneakyThrows: Error sneakyThrow(Throwable)>

   *
   * <p>We sometimes also use {@code sneakyThrow} for testing how our code responds to
   * sneaky checked exception.
   *
   * @return never; this method declares a return type of {@link Error} only so that callers can
   *     write {@code throw sneakyThrow(t);} to convince the compiler that the statement will always
   *     throw.
   */
  @CanIgnoreReturnValue
  static Error sneakyThrow(Throwable t) {
    throw new SneakyThrows<Error>().throwIt(t);
  }

  @SuppressWarnings("unchecked") // not really safe, but that's the point
  private Error throwIt(Throwable t) throws T {
    throw (T) t;
  }

  private SneakyThrows() {}
}
