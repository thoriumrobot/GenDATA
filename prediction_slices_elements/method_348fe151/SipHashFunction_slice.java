// Source-based slice around line 67
// Method: <com.google.common.hash.SipHashFunction: int bits()>

    checkArgument(
        d > 0, "The number of SipRound iterations (d=%s) during Finalization must be positive.", d);
    this.c = c;
    this.d = d;
    this.k0 = k0;
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

