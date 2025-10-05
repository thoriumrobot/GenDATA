// Source-based slice around line 45
// Method: com.google.common.primitives.UnsignedLong.ONE

 * @author Colin Evans
 * @since 11.0
 */
@GwtCompatible
public final class UnsignedLong extends Number implements Comparable<UnsignedLong> {

  private static final long UNSIGNED_MASK = 0x7fffffffffffffffL;

  public static final UnsignedLong ZERO = new UnsignedLong(0);
  public static final UnsignedLong ONE = new UnsignedLong(1);
  public static final UnsignedLong MAX_VALUE = new UnsignedLong(-1L);

  private final long value;

  private UnsignedLong(long value) {
    this.value = value;
  }

  /**
   * Returns an {@code UnsignedLong} corresponding to a given bit representation. The argument is
