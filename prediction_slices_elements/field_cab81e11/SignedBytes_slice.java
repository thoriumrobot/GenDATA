// Source-based slice around line 47
// Method: com.google.common.primitives.SignedBytes.MAX_POWER_OF_TWO

@GwtCompatible
public final class SignedBytes {
  private SignedBytes() {}

  /**
   * The largest power of two that can be represented as a signed {@code byte}.
   *
   * @since 10.0
   */
  public static final byte MAX_POWER_OF_TWO = 1 << 6;

  /**
   * Returns the {@code byte} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code byte} type
   * @return the {@code byte} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Byte#MAX_VALUE} or
   *     less than {@link Byte#MIN_VALUE}
   */
  public static byte checkedCast(long value) {
