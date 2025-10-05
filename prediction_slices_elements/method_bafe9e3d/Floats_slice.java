// Source-based slice around line 101
// Method: <com.google.common.primitives.Floats: boolean isFinite(float)>

  /**
   * Returns {@code true} if {@code value} represents a real number. This is equivalent to, but not
   * necessarily implemented as, {@code !(Float.isInfinite(value) || Float.isNaN(value))}.
   *
   * <p>Prefer {@link Float#isFinite(float)} instead.
   *
   * @since 10.0
   */
  @InlineMe(replacement = "Float.isFinite(value)")
  public static boolean isFinite(float value) {
    return Float.isFinite(value);
  }

  /**
   * Returns {@code true} if {@code target} is present as an element anywhere in {@code array}. Note
   * that this always returns {@code false} when {@code target} is {@code NaN}.
   *
   * @param array an array of {@code float} values, possibly empty
   * @param target a primitive {@code float} value
   * @return {@code true} if {@code array[i] == target} for some value of {@code i}
