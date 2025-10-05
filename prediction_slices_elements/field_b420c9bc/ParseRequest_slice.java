// Source-based slice around line 22
// Method: com.google.common.primitives.ParseRequest.rawValue

 */

package com.google.common.primitives;

import com.google.common.annotations.GwtCompatible;

/** A string to be parsed as a number and the radix to interpret it in. */
@GwtCompatible
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
