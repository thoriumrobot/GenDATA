// Source-based slice around line 41
// Method: com.google.common.escape.Platform.DEST_TL

    // requireNonNull accommodates Android's @RecentlyNullable annotation on ThreadLocal.get
    return requireNonNull(DEST_TL.get());
  }

  /**
   * A thread-local destination buffer to keep us from creating new buffers. The starting size is
   * 1024 characters. If we grow past this we don't put it back in the threadlocal, we just keep
   * going and grow as needed.
   */
  private static final ThreadLocal<char[]> DEST_TL =
      new ThreadLocal<char[]>() {
        @Override
        protected char[] initialValue() {
          return new char[1024];
        }
      };
}
