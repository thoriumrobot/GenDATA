// Source-based slice around line 288
// Method: <com.google.common.primitives.Floats: int checkNoOverflow(long)>

    float[] result = new float[checkNoOverflow(length)];
    int pos = 0;
    for (float[] array : arrays) {
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

  private static final class FloatConverter extends Converter<String, Float>
      implements Serializable {
    static final Converter<String, Float> INSTANCE = new FloatConverter();
