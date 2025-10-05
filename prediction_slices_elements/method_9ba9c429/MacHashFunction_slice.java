// Source-based slice around line 80
// Method: <com.google.common.hash.MacHashFunction: Hasher newHasher()>

      return mac;
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException(e);
    } catch (InvalidKeyException e) {
      throw new IllegalArgumentException(e);
    }
  }

  @Override
  public Hasher newHasher() {
    if (supportsClone) {
      try {
        return new MacHasher((Mac) prototype.clone());
      } catch (CloneNotSupportedException e) {
        // falls through
      }
    }
    return new MacHasher(getMac(prototype.getAlgorithm(), key));
  }

