// Source-based slice around line 291
// Method: <com.google.common.primitives.Doubles: int checkNoOverflow(long)>

    double[] result = new double[checkNoOverflow(length)];
    int pos = 0;
    for (double[] array : arrays) {
      System.arraycopy(array, 0, result, pos, array.length);
      pos += array.length;
    }
    return result;
  }

  private static int checkNoOverflow(long result) {
    checkArgument(
        result == (int) result,
        "the total number of elements (%s) in the arrays must fit in an int",
        result);
    return (int) result;
  }

  private static final class DoubleConverter extends Converter<String, Double>
      implements Serializable {
    static final Converter<String, Double> INSTANCE = new DoubleConverter();
