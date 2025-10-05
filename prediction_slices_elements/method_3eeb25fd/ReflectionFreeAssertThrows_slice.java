// Source-based slice around line 56
// Method: <com.google.common.testing.ReflectionFreeAssertThrows: T assertThrows(Class,ThrowingRunnable)>

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
          runnable.run();
          return null;
        },
        /* userPassedSupplier= */ false);
  }

