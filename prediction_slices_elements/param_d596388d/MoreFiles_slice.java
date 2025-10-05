// Source-based slice around line 743
// Method: <com.google.common.io.MoreFiles: void checkAllowsInsecure(Path,RecursiveDeleteOption[])>

      //      worrying about a symlink.
      return null;
    } else {
      // "foo" (working dir)
      return path.getFileSystem().getPath(".");
    }
  }

  /** Checks that the given options allow an insecure delete, throwing an exception if not. */
  private static void checkAllowsInsecure(Path path, RecursiveDeleteOption[] options)
      throws InsecureRecursiveDeleteException {
    if (!Arrays.asList(options).contains(RecursiveDeleteOption.ALLOW_INSECURE)) {
      throw new InsecureRecursiveDeleteException(path.toString());
    }
  }

  /**
   * Adds the given exception to the given collection, creating the collection if it's null. Returns
   * the collection.
   */
