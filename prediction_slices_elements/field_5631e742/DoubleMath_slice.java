// Source-based slice around line 323
// Method: com.google.common.math.DoubleMath.everySixteenthFactorial

        accum *= i;
      }
      return accum * everySixteenthFactorial[n >> 4];
    }
  }

  @VisibleForTesting static final int MAX_FACTORIAL = 170;

  @VisibleForTesting
  static final double[] everySixteenthFactorial = {
    0x1.0p0,
    0x1.30777758p44,
    0x1.956ad0aae33a4p117,
    0x1.ee69a78d72cb6p202,
    0x1.fe478ee34844ap295,
    0x1.c619094edabffp394,
    0x1.3638dd7bd6347p498,
    0x1.7cac197cfe503p605,
    0x1.1e5dfc140e1e5p716,
    0x1.8ce85fadb707ep829,
