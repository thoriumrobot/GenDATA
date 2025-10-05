// Source-based slice around line 94
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeChars(String)>

  }

  /**
   * Writes a {@code String} as specified by {@link DataOutputStream#writeChars(String)}, except
   * each character is written using little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
   */
  @Override
  public void writeChars(String s) throws IOException {
    for (int i = 0; i < s.length(); i++) {
      writeChar(s.charAt(i));
    }
  }

  /**
   * Writes a {@code double} as specified by {@link DataOutputStream#writeDouble(double)}, except
   * using little-endian byte order.
   *
   * @throws IOException if an I/O error occurs
