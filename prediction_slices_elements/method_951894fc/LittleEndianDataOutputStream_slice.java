// Source-based slice around line 118
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeFloat(float)>

  }

  /**
   * Writes a {@code float} as specified by {@link DataOutputStream#writeFloat(float)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeFloat(float v) throws IOException {
    writeInt(Float.floatToIntBits(v));
  }

  /**
   * Writes an {@code int} as specified by {@link DataOutputStream#writeInt(int)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
