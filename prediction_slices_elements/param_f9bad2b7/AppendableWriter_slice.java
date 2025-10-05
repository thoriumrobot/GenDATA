// Source-based slice around line 55
// Method: <com.google.common.io.AppendableWriter: void write(char[],int,int)>

  AppendableWriter(Appendable target) {
    this.target = checkNotNull(target);
  }

  /*
   * Abstract methods from Writer
   */

  @Override
  public void write(char[] cbuf, int off, int len) throws IOException {
    checkNotClosed();
    // It turns out that creating a new String is usually as fast, or faster
    // than wrapping cbuf in a light-weight CharSequence.
    target.append(new String(cbuf, off, len));
  }

  /*
   * Override a few functions for performance reasons to avoid creating unnecessary strings.
   */

