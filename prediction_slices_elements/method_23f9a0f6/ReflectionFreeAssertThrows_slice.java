// Source-based slice around line 50
// Method: <com.google.common.testing.ReflectionFreeAssertThrows: T assertThrows(Class,ThrowingSupplier)>

  interface ThrowingRunnable {
    void run() throws Throwable;
  }

  interface ThrowingSupplier {
    @Nullable Object get() throws Throwable;
  }

  @CanIgnoreReturnValue
  static <T extends Throwable> T assertThrows(
      Class<T> expectedThrowable, ThrowingSupplier supplier) {
    return doAssertThrows(expectedThrowable, supplier, /* userPassedSupplier= */ true);
  }

  @CanIgnoreReturnValue
  static <T extends Throwable> T assertThrows(
      Class<T> expectedThrowable, ThrowingRunnable runnable) {
    return doAssertThrows(
        expectedThrowable,
        () -> {
