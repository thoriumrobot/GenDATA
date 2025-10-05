// Source-based slice around line 57
// Method: <com.google.thirdparty.publicsuffix.PublicSuffixType: PublicSuffixType fromCode(char)>

  char getLeafNodeCode() {
    return leafNodeCode;
  }

  char getInnerNodeCode() {
    return innerNodeCode;
  }

  /** Returns a PublicSuffixType of the right type according to the given code */
  static PublicSuffixType fromCode(char code) {
    for (PublicSuffixType value : values()) {
      if (value.getInnerNodeCode() == code || value.getLeafNodeCode() == code) {
        return value;
      }
    }
    throw new IllegalArgumentException("No enum corresponding to given code: " + code);
  }
}
