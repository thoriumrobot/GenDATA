// Source-based slice around line 1115
// Method: <com.google.common.io.BaseEncoding: Writer separatingWriter(Writer,String,int)>

      @Override
      public Appendable append(@Nullable CharSequence chars) {
        throw new UnsupportedOperationException();
      }
    };
  }

  @J2ktIncompatible
  @GwtIncompatible // Writer
  static Writer separatingWriter(Writer delegate, String separator, int afterEveryChars) {
    Appendable separatingAppendable = separatingAppendable(delegate, separator, afterEveryChars);
    return new Writer() {
      @Override
      public void write(int c) throws IOException {
        separatingAppendable.append((char) c);
      }

      @Override
      public void write(char[] chars, int off, int len) throws IOException {
        throw new UnsupportedOperationException();
