// Source-based slice around line 30
// Method: <com.google.common.primitives.ParseRequest: ParseRequest fromString(String)>

final class ParseRequest {
  final String rawValue;
  final int radix;

  private ParseRequest(String rawValue, int radix) {
    this.rawValue = rawValue;
    this.radix = radix;
  }

  static ParseRequest fromString(String stringValue) {
    if (stringValue.length() == 0) {
      throw new NumberFormatException("empty string");
    }

    // Handle radix specifier if present
    String rawValue;
    int radix;
    char firstChar = stringValue.charAt(0);
    if (stringValue.startsWith("0x") || stringValue.startsWith("0X")) {
      rawValue = stringValue.substring(2);
