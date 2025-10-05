// Source-based slice around line 1083
// Method: <com.google.common.io.BaseEncoding: Appendable separatingAppendable(Appendable,String,int)>


      @Override
      public void close() throws IOException {
        delegate.close();
      }
    };
  }

  static Appendable separatingAppendable(
      Appendable delegate, String separator, int afterEveryChars) {
    checkNotNull(delegate);
    checkNotNull(separator);
    checkArgument(afterEveryChars > 0);
    return new Appendable() {
      int charsUntilSeparator = afterEveryChars;

      @Override
      public Appendable append(char c) throws IOException {
        if (charsUntilSeparator == 0) {
          delegate.append(separator);
