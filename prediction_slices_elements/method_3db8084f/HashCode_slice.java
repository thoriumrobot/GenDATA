// Source-based slice around line 246
// Method: <com.google.common.hash.HashCode: HashCode fromBytesNoCopy(byte[])>

  public static HashCode fromBytes(byte[] bytes) {
    checkArgument(bytes.length >= 1, "A HashCode must contain at least 1 byte.");
    return fromBytesNoCopy(bytes.clone());
  }

  /**
   * Creates a {@code HashCode} from a byte array. The array is <i>not</i> copied defensively, so it
   * must be handed-off so as to preserve the immutability contract of {@code HashCode}.
   */
  static HashCode fromBytesNoCopy(byte[] bytes) {
    return new BytesHashCode(bytes);
  }

  private static final class BytesHashCode extends HashCode implements Serializable {
    final byte[] bytes;

    BytesHashCode(byte[] bytes) {
      this.bytes = checkNotNull(bytes);
    }

