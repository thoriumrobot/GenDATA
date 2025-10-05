// Source-based slice around line 89
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger valueOf(BigInteger)>

    return fromIntBits((int) value);
  }

  /**
   * Returns a {@code UnsignedInteger} representing the same value as the specified {@link
   * BigInteger}. This is the inverse operation of {@link #bigIntegerValue()}.
   *
   * @throws IllegalArgumentException if {@code value} is negative or {@code value >= 2^32}
   */
  public static UnsignedInteger valueOf(BigInteger value) {
    checkNotNull(value);
    checkArgument(
        value.signum() >= 0 && value.bitLength() <= Integer.SIZE,
        "value (%s) is outside the range for an unsigned integer value",
        value);
    return fromIntBits(value.intValue());
  }

  /**
   * Returns an {@code UnsignedInteger} holding the value of the specified {@code String}, parsed as
