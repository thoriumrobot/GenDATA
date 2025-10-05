// Source-based slice around line 51
// Method: <com.google.common.base.SneakyThrows: Error throwIt(Throwable)>

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
