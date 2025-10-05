// Source-based slice around line 104
// Method: <com.google.common.math.MathPreconditions: void checkNoOverflow(boolean,String,long,long)>

    }
  }

  static void checkNoOverflow(boolean condition, String methodName, int a, int b) {
    if (!condition) {
      throw new ArithmeticException("overflow: " + methodName + "(" + a + ", " + b + ")");
    }
  }

  static void checkNoOverflow(boolean condition, String methodName, long a, long b) {
    if (!condition) {
      throw new ArithmeticException("overflow: " + methodName + "(" + a + ", " + b + ")");
    }
  }

  private MathPreconditions() {}
}
