// Source-based slice around line 107
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeDouble(double)>

  }

  /**
   * Writes a {@code double} as specified by {@link DataOutputStream#writeDouble(double)}, except
   * using little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeDouble(double v) throws IOException {
    writeLong(Double.doubleToLongBits(v));
  }

  /**
   * Writes a {@code float} as specified by {@link DataOutputStream#writeFloat(float)}, except using
   * little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
