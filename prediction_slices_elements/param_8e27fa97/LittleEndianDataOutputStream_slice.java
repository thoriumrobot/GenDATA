// Source-based slice around line 72
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeBytes(String)>

    ((DataOutputStream) out).writeByte(v);
  }

  /**
   * @deprecated The semantics of {@code writeBytes(String s)} are considered dangerous. Please use
   *     {@link #writeUTF(String s)}, {@link #writeChars(String s)} or another write method instead.
   */
  @Deprecated
  @Override
  public void writeBytes(String s) throws IOException {
    ((DataOutputStream) out).writeBytes(s);
  }

  /**
   * Writes a char as specified by {@link DataOutputStream#writeChar(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
