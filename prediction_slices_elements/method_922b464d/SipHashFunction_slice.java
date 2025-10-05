// Source-based slice around line 72
// Method: <com.google.common.hash.SipHashFunction: Hasher newHasher()>

    this.k1 = k1;
  }

  @Override
  public int bits() {
    return 64;
  }

  @Override
  public Hasher newHasher() {
    return new SipHasher(c, d, k0, k1);
  }

  // TODO(kak): Implement and benchmark the hashFoo() shortcuts.

  @Override
  public String toString() {
    return "Hashing.sipHash" + c + "" + d + "(" + k0 + ", " + k1 + ")";
  }

