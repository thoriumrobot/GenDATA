// Source-based slice around line 296
// Method: <com.google.common.math.IntMath: int sqrtFloor(int)>

         * halfSquare - sqrtFloor <= x < halfSquare + sqrtFloor + 1
         * so |x - halfSquare| <= sqrtFloor.  Therefore, it's safe to treat x - halfSquare as a
         * signed int, so lessThanBranchFree is safe for use.
         */
        return sqrtFloor + lessThanBranchFree(halfSquare, x);
    }
    throw new AssertionError();
  }

  private static int sqrtFloor(int x) {
    // There is no loss of precision in converting an int to a double, according to
    // http://java.sun.com/docs/books/jls/third_edition/html/conversions.html#5.1.2
    return (int) Math.sqrt(x);
  }

  /**
   * Returns the result of dividing {@code p} by {@code q}, rounding using the specified {@code
   * RoundingMode}. If the {@code RoundingMode} is {@link RoundingMode#DOWN}, then this method is
   * equivalent to regular Java division, {@code p / q}; and if it is {@link RoundingMode#FLOOR},
   * then this method is equivalent to {@link Math#floorDiv(int,int) Math.floorDiv}{@code (p, q)}.
