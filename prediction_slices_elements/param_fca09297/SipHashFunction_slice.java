// Source-based slice around line 84
// Method: <com.google.common.hash.SipHashFunction: boolean equals(Object)>


  // TODO(kak): Implement and benchmark the hashFoo() shortcuts.

  @Override
  public String toString() {
    return "Hashing.sipHash" + c + "" + d + "(" + k0 + ", " + k1 + ")";
  }

  @Override
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
