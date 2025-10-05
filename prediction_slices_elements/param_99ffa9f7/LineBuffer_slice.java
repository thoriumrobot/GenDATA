// Source-based slice around line 52
// Method: <com.google.common.io.LineBuffer: void add(char[],int,int)>

   * Process additional characters from the stream. When a line separator is found the contents of
   * the line and the line separator itself are passed to the abstract {@link #handleLine} method.
   *
   * @param cbuf the character buffer to process
   * @param off the offset into the buffer
   * @param len the number of characters to process
   * @throws IOException if an I/O error occurs
   * @see #finish
   */
  protected void add(char[] cbuf, int off, int len) throws IOException {
    int pos = off;
    if (sawReturn && len > 0) {
      // Last call to add ended with a CR; we can handle the line now.
      if (finishLine(cbuf[pos] == '\n')) {
        pos++;
      }
    }

    int start = pos;
    for (int end = off + len; pos < end; pos++) {
