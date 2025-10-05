// Source-based slice around line 129
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeInt(int)>

  }

  /**
   * Writes an {@code int} as specified by {@link DataOutputStream#writeInt(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeInt(int v) throws IOException {
    out.write(0xFF & v);
    out.write(0xFF & (v >> 8));
    out.write(0xFF & (v >> 16));
    out.write(0xFF & (v >> 24));
  }

  /**
   * Writes a {@code long} as specified by {@link DataOutputStream#writeLong(long)}, except using
   * little-endian byte order.
   *
