// Source-based slice around line 79
// Method: <com.google.common.hash.SipHashFunction: String toString()>


  @Override
  public Hasher newHasher() {
    return new SipHasher(c, d, k0, k1);
  }

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
