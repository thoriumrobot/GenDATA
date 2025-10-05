// Source-based slice around line 627
// Method: <com.google.common.math.Stats: Stats fromByteArray(byte[])>

  }

  /**
   * Creates a Stats instance from the given byte representation which was obtained by {@link
   * #toByteArray}.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   */
  public static Stats fromByteArray(byte[] byteArray) {
    checkNotNull(byteArray);
    checkArgument(
        byteArray.length == BYTES,
        "Expected Stats.BYTES = %s remaining , got %s",
        BYTES,
        byteArray.length);
    return readFrom(ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN));
  }

  /**
