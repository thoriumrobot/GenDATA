// Source-based slice around line 807
// Method: <com.google.common.io.MoreFiles: NoSuchFileException pathNotFound(Path,Collection)>

            null,
            "failed to delete one or more files; see suppressed exceptions for details");
    for (IOException e : exceptions) {
      deleteFailed.addSuppressed(e);
    }
    throw deleteFailed;
  }

  private static @Nullable NoSuchFileException pathNotFound(
      Path path, Collection<IOException> exceptions) {
    if (exceptions.size() != 1) {
      return null;
    }
    IOException exception = getOnlyElement(exceptions);
    if (!(exception instanceof NoSuchFileException)) {
      return null;
    }
    NoSuchFileException noSuchFileException = (NoSuchFileException) exception;
    String exceptionFile = noSuchFileException.getFile();
    if (exceptionFile == null) {
