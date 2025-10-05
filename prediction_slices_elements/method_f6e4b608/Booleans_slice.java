// Source-based slice around line 84
// Method: <com.google.common.primitives.Booleans: Comparator trueFirst()>


  /**
   * Returns a {@code Comparator<Boolean>} that sorts {@code true} before {@code false}.
   *
   * <p>This is particularly useful in Java 8+ in combination with {@code Comparator.comparing},
   * e.g. {@code Comparator.comparing(Foo::hasBar, trueFirst())}.
   *
   * @since 21.0
   */
  public static Comparator<Boolean> trueFirst() {
    return BooleanComparator.TRUE_FIRST;
  }

  /**
   * Returns a {@code Comparator<Boolean>} that sorts {@code false} before {@code true}.
   *
   * <p>This is particularly useful in Java 8+ in combination with {@code Comparator.comparing},
   * e.g. {@code Comparator.comparing(Foo::hasBar, falseFirst())}.
   *
   * @since 21.0
