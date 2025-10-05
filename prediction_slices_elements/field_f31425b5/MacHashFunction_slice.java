// Source-based slice around line 39
// Method: com.google.common.hash.MacHashFunction.key

 * @author Kurt Alfred Kluever
 */
@Immutable
final class MacHashFunction extends AbstractHashFunction {

  @SuppressWarnings("Immutable") // cloned before each use
  private final Mac prototype;

  @SuppressWarnings("Immutable") // keys are immutable, but not provably so
  private final Key key;

  private final String toString;
  private final int bits;
  private final boolean supportsClone;

  MacHashFunction(String algorithmName, Key key, String toString) {
    this.prototype = getMac(algorithmName, key);
    this.key = checkNotNull(key);
    this.toString = checkNotNull(toString);
    this.bits = prototype.getMacLength() * Byte.SIZE;
