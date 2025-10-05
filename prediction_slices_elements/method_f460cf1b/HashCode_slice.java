// Source-based slice around line 370
// Method: <com.google.common.hash.HashCode: boolean equals(Object)>


  /**
   * Returns {@code true} if {@code object} is a {@link HashCode} instance with the identical byte
   * representation to this hash code.
   *
   * <p><b>Security note:</b> this method uses a constant-time (not short-circuiting) implementation
   * to protect against <a href="http://en.wikipedia.org/wiki/Timing_attack">timing attacks</a>.
   */
  @Override
  public final boolean equals(@Nullable Object object) {
    if (object instanceof HashCode) {
      HashCode that = (HashCode) object;
      return bits() == that.bits() && equalsSameBits(that);
    }
    return false;
  }

  /**
   * Returns a "Java hash code" for this {@code HashCode} instance; this is well-defined (so, for
   * example, you can safely put {@code HashCode} instances into a {@code HashSet}) but is otherwise
