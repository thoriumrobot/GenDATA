// Source-based slice around line 48
// Method: com.google.common.primitives.UnsignedInteger.value

 * @author Louis Wasserman
 * @since 11.0
 */
@GwtCompatible
public final class UnsignedInteger extends Number implements Comparable<UnsignedInteger> {
  public static final UnsignedInteger ZERO = fromIntBits(0);
  public static final UnsignedInteger ONE = fromIntBits(1);
  public static final UnsignedInteger MAX_VALUE = fromIntBits(-1);

  private final int value;

  private UnsignedInteger(int value) {
    // GWT doesn't consistently overflow values to make them 32-bit, so we need to force it.
    this.value = value & 0xffffffff;
  }

  /**
   * Returns an {@code UnsignedInteger} corresponding to a given bit representation. The argument is
   * interpreted as an unsigned 32-bit value. Specifically, the sign bit of {@code bits} is
   * interpreted as a normal bit, and all other bits are treated as usual.
