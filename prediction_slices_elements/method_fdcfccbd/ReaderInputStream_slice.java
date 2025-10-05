// Source-based slice around line 203
// Method: <com.google.common.io.ReaderInputStream: void readMoreChars()>

  private static CharBuffer grow(CharBuffer buf) {
    char[] copy = Arrays.copyOf(buf.array(), buf.capacity() * 2);
    CharBuffer bigger = CharBuffer.wrap(copy);
    Java8Compatibility.position(bigger, buf.position());
    Java8Compatibility.limit(bigger, buf.limit());
    return bigger;
  }

  /** Handle the case of underflow caused by needing more input characters. */
  private void readMoreChars() throws IOException {
    // Possibilities:
    // 1) array has space available on right-hand side (between limit and capacity)
    // 2) array has space available on left-hand side (before position)
    // 3) array has no space available
    //
    // In case 2 we shift the existing chars to the left, and in case 3 we create a bigger
    // array, then they both become case 1.

    if (availableCapacity(charBuffer) == 0) {
      if (charBuffer.position() > 0) {
