// Source-based slice around line 314
// Method: <com.google.common.io.MoreFiles: Predicate isDirectory(LinkOption)>

      }
    }
    return ImmutableList.of();
  }

  /**
   * Returns a predicate that returns the result of {@link java.nio.file.Files#isDirectory(Path,
   * LinkOption...)} on input paths with the given link options.
   */
  public static Predicate<Path> isDirectory(LinkOption... options) {
    LinkOption[] optionsCopy = options.clone();
    return new Predicate<Path>() {
      @Override
      public boolean apply(Path input) {
        return Files.isDirectory(input, optionsCopy);
      }

      @Override
      public String toString() {
        return "MoreFiles.isDirectory(" + Arrays.toString(optionsCopy) + ")";
