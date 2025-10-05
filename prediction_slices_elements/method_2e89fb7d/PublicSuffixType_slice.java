// Source-based slice around line 52
// Method: <com.google.thirdparty.publicsuffix.PublicSuffixType: char getInnerNodeCode()>

  PublicSuffixType(char innerNodeCode, char leafNodeCode) {
    this.innerNodeCode = innerNodeCode;
    this.leafNodeCode = leafNodeCode;
  }

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
