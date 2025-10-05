// Source-based slice around line 123
// Method: com.google.common.testing.ReflectionFreeAssertThrows.INSTANCE_OF

      }
    };

    // used under GWT, etc., since the override of this method does not exist there
    ImmutableMap<Class<? extends Throwable>, Predicate<Throwable>> exceptions() {
      return ImmutableMap.of();
    }
  }

  private static final ImmutableMap<Class<? extends Throwable>, Predicate<Throwable>> INSTANCE_OF =
      ImmutableMap.<Class<? extends Throwable>, Predicate<Throwable>>builder()
          .put(ArithmeticException.class, e -> e instanceof ArithmeticException)
          .put(
              ArrayIndexOutOfBoundsException.class,
              e -> e instanceof ArrayIndexOutOfBoundsException)
          .put(ArrayStoreException.class, e -> e instanceof ArrayStoreException)
          .put(AssertionFailedError.class, e -> e instanceof AssertionFailedError)
          .put(CancellationException.class, e -> e instanceof CancellationException)
          .put(ClassCastException.class, e -> e instanceof ClassCastException)
          .put(
