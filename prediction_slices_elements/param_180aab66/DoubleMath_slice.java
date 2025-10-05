// Source-based slice around line 526
// Method: <com.google.common.math.DoubleMath: double checkFinite(double)>

      count++;
      // Art of Computer Programming vol. 2, Knuth, 4.2.2, (15)
      mean += (value - mean) / count;
    }
    return mean;
  }

  @GwtIncompatible // com.google.common.math.DoubleUtils
  @CanIgnoreReturnValue
  private static double checkFinite(double argument) {
    checkArgument(isFinite(argument));
    return argument;
  }

  private DoubleMath() {}
}
