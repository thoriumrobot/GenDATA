// Source-based slice around line 294
// Method: <com.google.common.collect.EnumMultiset: Iterator iterator()>

    checkNotNull(action);
    for (int i = 0; i < enumConstants.length; i++) {
      if (counts[i] > 0) {
        action.accept(enumConstants[i], counts[i]);
      }
    }
  }

  @Override
  public Iterator<E> iterator() {
    return Multisets.iteratorImpl(this);
  }

  @GwtIncompatible // java.io.ObjectOutputStream
  private void writeObject(ObjectOutputStream stream) throws IOException {
    stream.defaultWriteObject();
    stream.writeObject(type);
    Serialization.writeMultiset(this, stream);
  }

