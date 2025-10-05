// Source-based slice around line 143
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeLong(long)>

  }

  /**
   * Writes a {@code long} as specified by {@link DataOutputStream#writeLong(long)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeLong(long v) throws IOException {
    ((DataOutputStream) out).writeLong(Long.reverseBytes(v));
  }

  /**
   * Writes a {@code short} as specified by {@link DataOutputStream#writeShort(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
