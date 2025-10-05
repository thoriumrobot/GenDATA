// Source-based slice around line 67
// Method: <com.google.common.testing.ReflectionFreeAssertThrows: T doAssertThrows(Class,ThrowingSupplier,boolean)>

    return doAssertThrows(
        expectedThrowable,
        () -> {
          runnable.run();
          return null;
        },
        /* userPassedSupplier= */ false);
  }

  private static <T extends Throwable> T doAssertThrows(
      Class<T> expectedThrowable, ThrowingSupplier supplier, boolean userPassedSupplier) {
    checkNotNull(expectedThrowable);
    checkNotNull(supplier);
    Predicate<Throwable> predicate = INSTANCE_OF.get(expectedThrowable);
    if (predicate == null) {
      throw new IllegalArgumentException(
          expectedThrowable
              + " is not yet supported by ReflectionFreeAssertThrows. Add an entry for it in the"
              + " map in that class.");
    }
