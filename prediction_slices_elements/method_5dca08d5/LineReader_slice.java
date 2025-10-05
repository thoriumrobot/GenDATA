// Source-based slice around line 71
// Method: <com.google.common.io.LineReader: String readLine()>

   * Reads a line of text. A line is considered to be terminated by any one of a line feed ({@code
   * '\n'}), a carriage return ({@code '\r'}), or a carriage return followed immediately by a
   * linefeed ({@code "\r\n"}).
   *
   * @return a {@code String} containing the contents of the line, not including any
   *     line-termination characters, or {@code null} if the end of the stream has been reached.
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip a line
  public @Nullable String readLine() throws IOException {
    while (lines.peek() == null) {
      Java8Compatibility.clear(cbuf);
      // The default implementation of Reader#read(CharBuffer) allocates a
      // temporary char[], so we call Reader#read(char[], int, int) instead.
      int read = (reader != null) ? reader.read(buf, 0, buf.length) : readable.read(cbuf);
      if (read == -1) {
        lineBuf.finish();
        break;
      }
      lineBuf.add(buf, 0, read);
