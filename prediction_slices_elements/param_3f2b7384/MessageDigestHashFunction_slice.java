// Source-based slice around line 124
// Method: <com.google.common.hash.MessageDigestHashFunction: void readObject(ObjectInputStream)>

    }

    private static final long serialVersionUID = 0;
  }

  Object writeReplace() {
    return new SerializedForm(prototype.getAlgorithm(), bytes, toString);
  }

  private void readObject(ObjectInputStream stream) throws InvalidObjectException {
    throw new InvalidObjectException("Use SerializedForm");
  }

  /** Hasher that updates a message digest. */
  private static final class MessageDigestHasher extends AbstractByteHasher {
    private final MessageDigest digest;
    private final int bytes;
    private boolean done;

    private MessageDigestHasher(MessageDigest digest, int bytes) {
