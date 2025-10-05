// Source-based slice around line 53
// Method: <com.google.common.io.MultiReader: int read(char[],int,int)>

  /** Closes the current reader and opens the next one, if any. */
  private void advance() throws IOException {
    close();
    if (it.hasNext()) {
      current = it.next().openStream();
    }
  }

  @Override
  public int read(char[] cbuf, int off, int len) throws IOException {
    checkNotNull(cbuf);
    if (current == null) {
      return -1;
    }
    int result = current.read(cbuf, off, len);
    if (result == -1) {
      advance();
      return read(cbuf, off, len);
    }
    return result;
