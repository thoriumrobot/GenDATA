// Source-based slice around line 48
// Method: com.google.common.primitives.UnsignedLong.value

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
   * interpreted as an unsigned 64-bit value. Specifically, the sign bit of {@code bits} is
   * interpreted as a normal bit, and all other bits are treated as usual.
   *
