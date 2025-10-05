// Source-based slice around line 78
// Method: <com.google.common.io.CharSequenceReader: int read(CharBuffer)>

   * - Make checkOpen return the non-null `seq`. Then callers can assign that to a local variable or
   *   even back to `this.seq`. However, that may suggest that we're defending against concurrent
   *   mutation, which is not an actual risk because we use `synchronized`.
   * - Make `remaining` require a non-null `seq` argument. But this is a bit weird because the
   *   method, while it would avoid the instance field `seq` would still access the instance field
   *   `pos`.
   */

  @Override
  public synchronized int read(CharBuffer target) throws IOException {
    checkNotNull(target);
    checkOpen();
    requireNonNull(seq); // safe because of checkOpen
    if (!hasRemaining()) {
      return -1;
    }
    int charsToRead = min(target.remaining(), remaining());
    for (int i = 0; i < charsToRead; i++) {
      target.put(seq.charAt(pos++));
    }
