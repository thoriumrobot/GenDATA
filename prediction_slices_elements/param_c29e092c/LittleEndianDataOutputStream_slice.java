// Source-based slice around line 160
// Method: <com.google.common.io.LittleEndianDataOutputStream: void writeUTF(String)>

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
  // it silently ignores any exception thrown by flush(). Instead, just close the delegate stream.
  // It should flush itself if necessary.
  @Override
  public void close() throws IOException {
    out.close();
  }
