// Source-based slice around line 39
// Method: <com.google.common.hash.HashCode: int bits()>

 *
 * @author Dimitris Andreou
 * @author Kurt Alfred Kluever
 * @since 11.0
 */
public abstract class HashCode {
  HashCode() {}

  /** Returns the number of bits in this hash code; a positive multiple of 8. */
  public abstract int bits();

  /**
   * Returns the first four bytes of {@linkplain #asBytes() this hashcode's bytes}, converted to an
   * {@code int} value in little-endian order.
   *
   * @throws IllegalStateException if {@code bits() < 32}
   */
  public abstract int asInt();

  /**
