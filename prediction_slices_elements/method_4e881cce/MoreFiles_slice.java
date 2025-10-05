// Source-based slice around line 754
// Method: <com.google.common.io.MoreFiles: Collection addException(Collection,IOException)>

    if (!Arrays.asList(options).contains(RecursiveDeleteOption.ALLOW_INSECURE)) {
      throw new InsecureRecursiveDeleteException(path.toString());
    }
  }

  /**
   * Adds the given exception to the given collection, creating the collection if it's null. Returns
   * the collection.
   */
  private static Collection<IOException> addException(
      @Nullable Collection<IOException> exceptions, IOException e) {
    if (exceptions == null) {
      exceptions = new ArrayList<>(); // don't need Set semantics
    }
    exceptions.add(e);
    return exceptions;
  }

  /**
   * Concatenates the contents of the two given collections of exceptions. If either collection is
