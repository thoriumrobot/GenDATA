// Source-based slice around line 72
// Method: <com.google.common.hash.HashCode: byte[] asBytes()>

   */
  public abstract long padToLong();

  /**
   * Returns the value of this hash code as a byte array. The caller may modify the byte array;
   * changes to it will <i>not</i> be reflected in this {@code HashCode} object or any other arrays
   * returned by this method.
   */
  // TODO(user): consider ByteString here, when that is available
  public abstract byte[] asBytes();

  /**
   * Copies bytes from this hash code into {@code dest}.
   *
   * @param dest the byte array into which the hash code will be written
   * @param offset the start offset in the data
   * @param maxLength the maximum number of bytes to write
   * @return the number of bytes written to {@code dest}
   * @throws IndexOutOfBoundsException if there is not enough room in {@code dest}
   */
