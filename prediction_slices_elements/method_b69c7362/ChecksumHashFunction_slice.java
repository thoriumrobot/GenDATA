// Source-based slice around line 56
// Method: <com.google.common.hash.ChecksumHashFunction: Hasher newHasher()>

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
    return toString;
  }

  /** Hasher that updates a checksum. */
  private final class ChecksumHasher extends AbstractByteHasher {
