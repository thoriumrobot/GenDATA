// Source-based slice around line 98
// Method: <com.google.common.hash.HashCode: byte[] getBytesInternal()>

  }

  abstract void writeBytesToImpl(byte[] dest, int offset, int maxLength);

  /**
   * Returns a mutable view of the underlying bytes for the given {@code HashCode} if it is a
   * byte-based hashcode. Otherwise it returns {@link HashCode#asBytes}. Do <i>not</i> mutate this
   * array or else you will break the immutability contract of {@code HashCode}.
   */
  byte[] getBytesInternal() {
    return asBytes();
  }

  /**
   * Returns whether this {@code HashCode} and that {@code HashCode} have the same value, given that
   * they have the same number of bits.
   */
  abstract boolean equalsSameBits(HashCode that);

  /**
