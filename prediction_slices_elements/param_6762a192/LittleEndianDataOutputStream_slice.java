// Source-based slice around line 83
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeChar(int)>

  }

  /**
   * Writes a char as specified by {@link DataOutputStream#writeChar(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeChar(int v) throws IOException {
    writeShort(v);
  }

  /**
   * Writes a {@code String} as specified by {@link DataOutputStream#writeChars(String)}, except
   * each character is written using little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
