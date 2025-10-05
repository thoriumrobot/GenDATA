// Source-based slice around line 631
// Method: <com.google.common.base.Ascii: int getAlphaIndex(char)>

      return false;
    }
    return true;
  }

  /**
   * Returns the non-negative index value of the alpha character {@code c}, regardless of case. Ie,
   * 'a'/'A' returns 0 and 'z'/'Z' returns 25. Non-alpha characters return a value of 26 or greater.
   */
  private static int getAlphaIndex(char c) {
    // Fold upper-case ASCII to lower-case and make zero-indexed and unsigned (by casting to char).
    return (char) ((c | CASE_MASK) - 'a');
  }
}
