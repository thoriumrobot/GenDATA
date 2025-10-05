// Source-based slice around line 93
// Method: <com.google.common.hash.SipHashFunction: int hashCode()>

  public boolean equals(@Nullable Object object) {
    if (object instanceof SipHashFunction) {
      SipHashFunction other = (SipHashFunction) object;
      return (c == other.c) && (d == other.d) && (k0 == other.k0) && (k1 == other.k1);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return (int) (getClass().hashCode() ^ c ^ d ^ k0 ^ k1);
  }

  private static final class SipHasher extends AbstractStreamingHasher {
    private static final int CHUNK_SIZE = 8;

    // The number of compression rounds.
    private final int c;
    // The number of finalization rounds.
    private final int d;
