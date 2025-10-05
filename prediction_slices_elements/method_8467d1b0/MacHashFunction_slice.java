// Source-based slice around line 92
// Method: <com.google.common.hash.MacHashFunction: String toString()>

        return new MacHasher((Mac) prototype.clone());
      } catch (CloneNotSupportedException e) {
        // falls through
      }
    }
    return new MacHasher(getMac(prototype.getAlgorithm(), key));
  }

  @Override
  public String toString() {
    return toString;
  }

  /** Hasher that updates a {@link Mac} (message authentication code). */
  private static final class MacHasher extends AbstractByteHasher {
    private final Mac mac;
    private boolean done;

    private MacHasher(Mac mac) {
      this.mac = mac;
