/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.math;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.value.qual.PolyValue;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import static java.math.BigInteger.LONG_MASK;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Objects;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public class BigDecimal extends Number implements Comparable<BigDecimal> {

    @Positive
    public static final BigDecimal ZERO;

    @Positive
    public static final BigDecimal ONE;

    @Positive
    public static final BigDecimal TEN;

    @Positive
    public BigDecimal(char[] in, int offset, int len) {
    @Positive
    }

    @Positive
    public BigDecimal(char[] in, int offset, int len, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(char[] in) {
    @Positive
    }

    @Positive
    public BigDecimal(char[] in, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(String val) {
    @Positive
    }

    @Positive
    public BigDecimal(String val, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(double val) {
    @Positive
    }

    @Positive
    public BigDecimal(double val, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(BigInteger val) {
    @Positive
    }

    @Positive
    public BigDecimal(BigInteger val, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(BigInteger unscaledVal, int scale) {
    @Positive
    }

    @Positive
    public BigDecimal(BigInteger unscaledVal, int scale, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(int val) {
    @Positive
    }

    @Positive
    public BigDecimal(int val, MathContext mc) {
    @Positive
    }

    @Positive
    public BigDecimal(long val) {
    @Positive
    }

    @Positive
    public BigDecimal(long val, MathContext mc) {
    @Positive
    }

    @Positive
    public static BigDecimal valueOf(long unscaledVal, int scale);

    @Positive
    public static BigDecimal valueOf(long val);

    @Positive
    static BigDecimal valueOf(long unscaledVal, int scale, int prec);

    @Positive
    static BigDecimal valueOf(BigInteger intVal, int scale, int prec);

    @Positive
    static BigDecimal zeroValueOf(int scale);

    @Positive
    public static BigDecimal valueOf(double val);

    @Positive
    public BigDecimal add(BigDecimal augend);

    @Positive
    public BigDecimal add(BigDecimal augend, MathContext mc);

    @Positive
    public BigDecimal subtract(BigDecimal subtrahend);

    @Positive
    public BigDecimal subtract(BigDecimal subtrahend, MathContext mc);

    @Positive
    public BigDecimal multiply(BigDecimal multiplicand);

    @Positive
    public BigDecimal multiply(BigDecimal multiplicand, MathContext mc);

    @Positive
    @Deprecated()
    @Positive
    public BigDecimal divide(BigDecimal divisor, int scale, int roundingMode);

    @Positive
    public BigDecimal divide(BigDecimal divisor, int scale, RoundingMode roundingMode);

    @Positive
    @Deprecated()
    @Positive
    public BigDecimal divide(BigDecimal divisor, int roundingMode);

    @Positive
    public BigDecimal divide(BigDecimal divisor, RoundingMode roundingMode);

    @Positive
    public BigDecimal divide(BigDecimal divisor);

    @Positive
    public BigDecimal divide(BigDecimal divisor, MathContext mc);

    @Positive
    public BigDecimal divideToIntegralValue(BigDecimal divisor);

    @Positive
    public BigDecimal divideToIntegralValue(BigDecimal divisor, MathContext mc);

    @Positive
    public BigDecimal remainder(BigDecimal divisor);

    @Positive
    public BigDecimal remainder(BigDecimal divisor, MathContext mc);

    @Positive
    public BigDecimal[] divideAndRemainder(BigDecimal divisor);

    @Positive
    public BigDecimal[] divideAndRemainder(BigDecimal divisor, MathContext mc);

    @Positive
    public BigDecimal sqrt(MathContext mc);

    @Positive
    public BigDecimal pow(int n);

    @Positive
    public BigDecimal pow(int n, MathContext mc);

    @Positive
    public BigDecimal abs();

    @Positive
    public BigDecimal abs(MathContext mc);

    @Positive
    public BigDecimal negate();

    @Positive
    public BigDecimal negate(MathContext mc);

    @Positive
    public BigDecimal plus();

    @Positive
    public BigDecimal plus(MathContext mc);

    @Positive
    public int signum();

    @Positive
    public int scale();

    @Positive
    public int precision();

    @Positive
    public BigInteger unscaledValue();

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_UP;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_DOWN;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_CEILING;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_FLOOR;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_HALF_UP;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_HALF_DOWN;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_HALF_EVEN;

    @Positive
    @Deprecated()
    @Positive
    public static final int ROUND_UNNECESSARY;

    @Positive
    public BigDecimal round(MathContext mc);

    @Positive
    public BigDecimal setScale(int newScale, RoundingMode roundingMode);

    @Positive
    @Deprecated()
    @Positive
    public BigDecimal setScale(int newScale, int roundingMode);

    @Positive
    public BigDecimal setScale(int newScale);

    @Positive
    public BigDecimal movePointLeft(int n);

    @Positive
    public BigDecimal movePointRight(int n);

    @Positive
    public BigDecimal scaleByPowerOfTen(int n);

    @Positive
    public BigDecimal stripTrailingZeros();

    @Positive
    @Override
    @Positive
    public int compareTo(BigDecimal val);

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object x);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public BigDecimal min(BigDecimal val);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    public BigDecimal max(BigDecimal val);

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    public String toEngineeringString();

    @Positive
    public String toPlainString();

    @Positive
    public BigInteger toBigInteger();

    @Positive
    public BigInteger toBigIntegerExact();

    @Positive
    @Override
    @Positive
    @PolyValue
    @Positive
    public long longValue(@PolyValue BigDecimal this);

    @Positive
    public long longValueExact();

    @Positive
    private static class LongOverflow {

    @Positive
        public static void check(BigDecimal num);
    @Positive
    }

    @Positive
    @Override
    @Positive
    @PolyValue
    @Positive
    public int intValue(@PolyValue BigDecimal this);

    @Positive
    public int intValueExact();

    @Positive
    public short shortValueExact();

    @Positive
    public byte byteValueExact();

    @Positive
    @Override
    @Positive
    @PolyValue
    @Positive
    public float floatValue(@PolyValue BigDecimal this);

    @Positive
    @Override
    @Positive
    @PolyValue
    @Positive
    public double doubleValue(@PolyValue BigDecimal this);

    @Positive
    public BigDecimal ulp();

    @Positive
    static class StringBuilderHelper {

    @Positive
        StringBuilder getStringBuilder();

    @Positive
        char[] getCompactCharArray();

    @Positive
        int putIntCompact(long intCompact);
    @Positive
    }

    @Positive
    private static class UnsafeHolder {

    @Positive
        static void setIntCompact(BigDecimal bd, long val);

    @Positive
        static void setIntValVolatile(BigDecimal bd, BigInteger val);
    @Positive
    }

    @Positive
    static int longDigitLength(long x);

    @Positive
    static BigDecimal scaledTenPow(int n, int sign, int scale);
    @Positive
}

// CFWR semantic augmentation - variant 0
