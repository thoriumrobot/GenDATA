// Source-based slice around line 45
// Method: <com.google.common.io.MultiReader: void advance()>

  private final Iterator<? extends CharSource> it;
  private @Nullable Reader current;

  MultiReader(Iterator<? extends CharSource> readers) throws IOException {
    this.it = readers;
    advance();
  }

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
