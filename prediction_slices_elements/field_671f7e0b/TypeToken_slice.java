// Source-based slice around line 1470
// Method: com.google.common.reflect.TypeToken.serialVersionUID

      @Override
      @Nullable K getSuperclass(K type) {
        return delegate.getSuperclass(type);
      }
    }
  }

  // This happens to be the hash of the class as of now. So setting it makes a backward compatible
  // change. Going forward, if any incompatible change is added, we can change the UID back to 1.
  private static final long serialVersionUID = 3637540370352322684L;
}
