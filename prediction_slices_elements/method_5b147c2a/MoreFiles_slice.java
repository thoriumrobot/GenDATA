// Source-based slice around line 768
// Method: <com.google.common.io.MoreFiles: Collection concat(Collection,Collection)>

    exceptions.add(e);
    return exceptions;
  }

  /**
   * Concatenates the contents of the two given collections of exceptions. If either collection is
   * null, the other collection is returned. Otherwise, the elements of {@code other} are added to
   * {@code exceptions} and {@code exceptions} is returned.
   */
  private static @Nullable Collection<IOException> concat(
      @Nullable Collection<IOException> exceptions, @Nullable Collection<IOException> other) {
    if (exceptions == null) {
      return other;
    } else if (other != null) {
      exceptions.addAll(other);
    }
    return exceptions;
  }

  /**
