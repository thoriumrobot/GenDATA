// Source-based slice around line 70
// Method: <com.google.common.io.MultiInputStream: int available()>

  /** Closes the current input stream and opens the next one, if any. */
  private void advance() throws IOException {
    close();
    if (it.hasNext()) {
      in = it.next().openStream();
    }
  }

  @Override
  public int available() throws IOException {
    if (in == null) {
      return 0;
    }
    return in.available();
  }

  @Override
  public boolean markSupported() {
    return false;
  }
