// Source-based slice around line 230
// Method: <com.google.common.io.LittleEndianDataInputStream: byte readAndCheckByte()>


  /**
   * Reads a byte from the input stream checking that the end of file (EOF) has not been
   * encountered.
   *
   * @return byte read from input
   * @throws IOException if an error is encountered while reading
   * @throws EOFException if the end of file (EOF) is encountered.
   */
  private byte readAndCheckByte() throws IOException, EOFException {
    int b1 = in.read();

    if (b1 == -1) {
      throw new EOFException();
    }

    return (byte) b1;
  }
}
