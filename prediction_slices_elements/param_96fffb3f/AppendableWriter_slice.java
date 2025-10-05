// Source-based slice around line 67
// Method: <com.google.common.io.AppendableWriter: void write(int)>

    // than wrapping cbuf in a light-weight CharSequence.
    target.append(new String(cbuf, off, len));
  }

  /*
   * Override a few functions for performance reasons to avoid creating unnecessary strings.
   */

  @Override
  public void write(int c) throws IOException {
    checkNotClosed();
    target.append((char) c);
  }

  @Override
  public void write(String str) throws IOException {
    checkNotNull(str);
    checkNotClosed();
    target.append(str);
  }
