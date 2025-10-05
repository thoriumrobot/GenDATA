// Source-based slice around line 62
// Method: <com.google.common.io.MultiInputStream: void advance()>

      try {
        in.close();
      } finally {
        in = null;
      }
    }
  }

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
