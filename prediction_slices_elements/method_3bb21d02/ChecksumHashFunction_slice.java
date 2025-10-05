// Source-based slice around line 51
// Method: <com.google.common.hash.ChecksumHashFunction: int bits()>

  ChecksumHashFunction(
      ImmutableSupplier<? extends Checksum> checksumSupplier, int bits, String toString) {
    this.checksumSupplier = checkNotNull(checksumSupplier);
    checkArgument(bits == 32 || bits == 64, "bits (%s) must be either 32 or 64", bits);
    this.bits = bits;
    this.toString = checkNotNull(toString);
  }

  @Override
  public int bits() {
    return bits;
  }

  @Override
  public Hasher newHasher() {
    return new ChecksumHasher(checksumSupplier.get());
  }

  @Override
  public String toString() {
