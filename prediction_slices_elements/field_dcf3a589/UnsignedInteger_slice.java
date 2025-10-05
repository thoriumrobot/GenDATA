// Source-based slice around line 46
// Method: com.google.common.primitives.UnsignedInteger.MAX_VALUE

 * primitive utilities</a>.
 *
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
