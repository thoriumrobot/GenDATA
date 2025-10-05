// Source-based slice around line 44
// Method: com.google.common.hash.MessageDigestHashFunction.toString

 */
@Immutable
final class MessageDigestHashFunction extends AbstractHashFunction implements Serializable {

  @SuppressWarnings("Immutable") // cloned before each use
  private final MessageDigest prototype;

  private final int bytes;
  private final boolean supportsClone;
  private final String toString;

  MessageDigestHashFunction(String algorithmName, String toString) {
    this.prototype = getMessageDigest(algorithmName);
    this.bytes = prototype.getDigestLength();
    this.toString = checkNotNull(toString);
    this.supportsClone = supportsClone(prototype);
  }

  MessageDigestHashFunction(String algorithmName, int bytes, String toString) {
    this.toString = checkNotNull(toString);
