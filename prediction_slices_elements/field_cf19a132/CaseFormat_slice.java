// Source-based slice around line 117
// Method: com.google.common.base.CaseFormat.wordSeparator

      }
      if (format == LOWER_UNDERSCORE) {
        return Ascii.toLowerCase(s);
      }
      return super.convert(format, s);
    }
  };

  private final CharMatcher wordBoundary;
  private final String wordSeparator;

  CaseFormat(CharMatcher wordBoundary, String wordSeparator) {
    this.wordBoundary = wordBoundary;
    this.wordSeparator = wordSeparator;
  }

  /**
   * Converts the specified {@code String str} from this format to the specified {@code format}. A
   * "best effort" approach is taken; if {@code str} does not conform to the assumed format, then
   * the behavior of this method is undefined but we make a reasonable effort at converting anyway.
