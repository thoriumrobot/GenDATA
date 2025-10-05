// Source-based slice around line 65
// Method: com.google.common.primitives.UnsignedBytes.MAX_POWER_OF_TWO

@GwtIncompatible
public final class UnsignedBytes {
  private UnsignedBytes() {}

  /**
   * The largest power of two that can be represented as an unsigned {@code byte}.
   *
   * @since 10.0
   */
  public static final byte MAX_POWER_OF_TWO = (byte) 0x80;

  /**
   * The largest value that fits into an unsigned byte.
   *
   * @since 13.0
   */
  public static final byte MAX_VALUE = (byte) 0xFF;

  private static final int UNSIGNED_MASK = 0xFF;

