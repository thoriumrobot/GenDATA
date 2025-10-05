// Source-based slice around line 60
// Method: <com.google.common.io.CharSequenceReader: int remaining()>

    if (seq == null) {
      throw new IOException("reader closed");
    }
  }

  private boolean hasRemaining() {
    return remaining() > 0;
  }

  private int remaining() {
    requireNonNull(seq); // safe as long as we call this only after checkOpen
    return seq.length() - pos;
  }

  /*
   * To avoid the need to call requireNonNull so much, we could consider more clever approaches,
   * such as:
   *
   * - Make checkOpen return the non-null `seq`. Then callers can assign that to a local variable or
   *   even back to `this.seq`. However, that may suggest that we're defending against concurrent
