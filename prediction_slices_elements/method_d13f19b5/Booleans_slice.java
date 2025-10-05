// Source-based slice around line 96
// Method: <com.google.common.primitives.Booleans: Comparator falseFirst()>


  /**
   * Returns a {@code Comparator<Boolean>} that sorts {@code false} before {@code true}.
   *
   * <p>This is particularly useful in Java 8+ in combination with {@code Comparator.comparing},
   * e.g. {@code Comparator.comparing(Foo::hasBar, falseFirst())}.
   *
   * @since 21.0
   */
  public static Comparator<Boolean> falseFirst() {
    return BooleanComparator.FALSE_FIRST;
  }

  /**
   * Returns a hash code for {@code value}; obsolete alternative to {@link
   * Boolean#hashCode(boolean)}.
   *
   * @param value a primitive {@code boolean} value
   * @return a hash code for the value
   */
