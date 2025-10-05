// Source-based slice around line 105
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger valueOf(String)>

  }

  /**
   * Returns an {@code UnsignedInteger} holding the value of the specified {@code String}, parsed as
   * an unsigned {@code int} value.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code int}
   *     value
   */
  public static UnsignedInteger valueOf(String string) {
    return valueOf(string, 10);
  }

  /**
   * Returns an {@code UnsignedInteger} holding the value of the specified {@code String}, parsed as
   * an unsigned {@code int} value in the specified radix.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code int}
   *     value
   */
