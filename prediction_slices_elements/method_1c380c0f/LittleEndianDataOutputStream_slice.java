// Source-based slice around line 154
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeShort(int)>

  }

  /**
   * Writes a {@code short} as specified by {@link DataOutputStream#writeShort(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeShort(int v) throws IOException {
    out.write(0xFF & v);
    out.write(0xFF & (v >> 8));
  }

  @Override
  public void writeUTF(String str) throws IOException {
    ((DataOutputStream) out).writeUTF(str);
  }

  // Overriding close() because FilterOutputStream's close() method pre-JDK8 has bad behavior:
