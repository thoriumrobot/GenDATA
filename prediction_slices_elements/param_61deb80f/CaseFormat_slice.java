// Source-based slice around line 212
// Method: <com.google.common.base.CaseFormat: String normalizeFirstWord(String)>

    public String toString() {
      return sourceFormat + ".converterTo(" + targetFormat + ")";
    }

    @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0L;
  }

  abstract String normalizeWord(String word);

  String normalizeFirstWord(String word) {
    return normalizeWord(word);
  }

  private static String firstCharOnlyToUpper(String word) {
    return word.isEmpty()
        ? word
        : Ascii.toUpperCase(word.charAt(0)) + Ascii.toLowerCase(word.substring(1));
  }
}
