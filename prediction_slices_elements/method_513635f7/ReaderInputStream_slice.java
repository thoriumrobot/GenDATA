// Source-based slice around line 194
// Method: <com.google.common.io.ReaderInputStream: CharBuffer grow(CharBuffer)>

          // Only reach here if a CharsetEncoder with non-REPLACE settings is used.
          result.throwException();
          return 0; // Not called.
        }
      }
    }
  }

  /** Returns a new CharBuffer identical to buf, except twice the capacity. */
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
