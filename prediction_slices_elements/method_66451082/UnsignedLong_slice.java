// Source-based slice around line 91
// Method: <com.google.common.primitives.UnsignedLong: UnsignedLong valueOf(BigInteger)>

  }

  /**
   * Returns a {@code UnsignedLong} representing the same value as the specified {@code BigInteger}.
   * This is the inverse operation of {@link #bigIntegerValue()}.
   *
   * @throws IllegalArgumentException if {@code value} is negative or {@code value >= 2^64}
   */
  @CanIgnoreReturnValue
  public static UnsignedLong valueOf(BigInteger value) {
    checkNotNull(value);
    checkArgument(
        value.signum() >= 0 && value.bitLength() <= Long.SIZE,
        "value (%s) is outside the range for an unsigned long value",
        value);
    return fromLongBits(value.longValue());
  }

  /**
   * Returns an {@code UnsignedLong} holding the value of the specified {@code String}, parsed as an
