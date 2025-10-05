// Source-based slice around line 1057
// Method: <com.google.common.io.BaseEncoding: Reader ignoringReader(Reader,String)>


    @Override
    BaseEncoding newInstance(Alphabet alphabet, @Nullable Character paddingChar) {
      return new Base64Encoding(alphabet, paddingChar);
    }
  }

  @J2ktIncompatible
  @GwtIncompatible
  static Reader ignoringReader(Reader delegate, String toIgnore) {
    checkNotNull(delegate);
    checkNotNull(toIgnore);
    return new Reader() {
      @Override
      public int read() throws IOException {
        int readChar;
        do {
          readChar = delegate.read();
        } while (readChar != -1 && toIgnore.indexOf((char) readChar) >= 0);
        return readChar;
