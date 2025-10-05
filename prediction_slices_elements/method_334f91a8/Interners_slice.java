// Source-based slice around line 165
// Method: <com.google.common.collect.Interners: Function asFunction(Interner)>

      }
    }
  }

  /**
   * Returns a function that delegates to the {@link Interner#intern} method of the given interner.
   *
   * @since 8.0
   */
  public static <E> Function<E, E> asFunction(Interner<E> interner) {
    return new InternerFunction<>(checkNotNull(interner));
  }

  private static final class InternerFunction<E> implements Function<E, E> {

    private final Interner<E> interner;

    InternerFunction(Interner<E> interner) {
      this.interner = interner;
    }
