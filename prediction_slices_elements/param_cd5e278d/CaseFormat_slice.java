// Source-based slice around line 129
// Method: <com.google.common.base.CaseFormat: String to(CaseFormat,String)>

    this.wordBoundary = wordBoundary;
    this.wordSeparator = wordSeparator;
  }

  /**
   * Converts the specified {@code String str} from this format to the specified {@code format}. A
   * "best effort" approach is taken; if {@code str} does not conform to the assumed format, then
   * the behavior of this method is undefined but we make a reasonable effort at converting anyway.
   */
  public final String to(CaseFormat format, String str) {
    checkNotNull(format);
    checkNotNull(str);
    return (format == this) ? str : convert(format, str);
  }

  /** Enum values can override for performance reasons. */
  String convert(CaseFormat format, String s) {
    // deal with camel conversion
    StringBuilder out = null;
    int i = 0;
