// Source-based slice around line 237
// Method: <com.google.common.hash.HashCode: HashCode fromBytes(byte[])>

    private static final long serialVersionUID = 0;
  }

  /**
   * Creates a {@code HashCode} from a byte array. The array is defensively copied to preserve the
   * immutability contract of {@code HashCode}. The array cannot be empty.
   *
   * @since 15.0 (since 12.0 in HashCodes)
   */
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
