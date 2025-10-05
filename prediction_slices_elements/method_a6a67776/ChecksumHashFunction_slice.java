// Source-based slice around line 61
// Method: <com.google.common.hash.ChecksumHashFunction: String toString()>

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
    private final Checksum checksum;

    private ChecksumHasher(Checksum checksum) {
      this.checksum = checkNotNull(checksum);
    }
