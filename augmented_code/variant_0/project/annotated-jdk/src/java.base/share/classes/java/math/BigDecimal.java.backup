/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.math;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.value.qual.PolyValue;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import static java.math.BigInteger.LONG_MASK;
import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

@AnnotatedFor("nullness")
public class BigDecimal extends Number implements Comparable<BigDecimal> {

    public static final BigDecimal ZERO;

    public static final BigDecimal ONE;

    public static final BigDecimal TEN;

    public BigDecimal(char[] in, int offset, int len) {
    }

    public BigDecimal(char[] in, int offset, int len, MathContext mc) {
    }

    public BigDecimal(char[] in) {
    }

    public BigDecimal(char[] in, MathContext mc) {
    }

    public BigDecimal(String val) {
    }

    public BigDecimal(String val, MathContext mc) {
    }

    public BigDecimal(double val) {
    }

    public BigDecimal(double val, MathContext mc) {
    }

    public BigDecimal(BigInteger val) {
    }

    public BigDecimal(BigInteger val, MathContext mc) {
    }

    public BigDecimal(BigInteger unscaledVal, int scale) {
    }

    public BigDecimal(BigInteger unscaledVal, int scale, MathContext mc) {
    }

    public BigDecimal(int val) {
    }

    public BigDecimal(int val, MathContext mc) {
    }

    public BigDecimal(long val) {
    }

    public BigDecimal(long val, MathContext mc) {
    }

    public static BigDecimal valueOf(long unscaledVal, int scale);

    public static BigDecimal valueOf(long val);

    static BigDecimal valueOf(long unscaledVal, int scale, int prec);

    static BigDecimal valueOf(BigInteger intVal, int scale, int prec);

    static BigDecimal zeroValueOf(int scale);

    public static BigDecimal valueOf(double val);

    public BigDecimal add(BigDecimal augend);

    public BigDecimal add(BigDecimal augend, MathContext mc);

    public BigDecimal subtract(BigDecimal subtrahend);

    public BigDecimal subtract(BigDecimal subtrahend, MathContext mc);

    public BigDecimal multiply(BigDecimal multiplicand);

    public BigDecimal multiply(BigDecimal multiplicand, MathContext mc);

    @Deprecated()
    public BigDecimal divide(BigDecimal divisor, int scale, int roundingMode);

    public BigDecimal divide(BigDecimal divisor, int scale, RoundingMode roundingMode);

    @Deprecated()
    public BigDecimal divide(BigDecimal divisor, int roundingMode);

    public BigDecimal divide(BigDecimal divisor, RoundingMode roundingMode);

    public BigDecimal divide(BigDecimal divisor);

    public BigDecimal divide(BigDecimal divisor, MathContext mc);

    public BigDecimal divideToIntegralValue(BigDecimal divisor);

    public BigDecimal divideToIntegralValue(BigDecimal divisor, MathContext mc);

    public BigDecimal remainder(BigDecimal divisor);

    public BigDecimal remainder(BigDecimal divisor, MathContext mc);

    public BigDecimal[] divideAndRemainder(BigDecimal divisor);

    public BigDecimal[] divideAndRemainder(BigDecimal divisor, MathContext mc);

    public BigDecimal sqrt(MathContext mc);

    public BigDecimal pow(int n);

    public BigDecimal pow(int n, MathContext mc);

    public BigDecimal abs();

    public BigDecimal abs(MathContext mc);

    public BigDecimal negate();

    public BigDecimal negate(MathContext mc);

    public BigDecimal plus();

    public BigDecimal plus(MathContext mc);

    public int signum();

    public int scale();

    public int precision();

    public BigInteger unscaledValue();

    @Deprecated()
    public static final int ROUND_UP;

    @Deprecated()
    public static final int ROUND_DOWN;

    @Deprecated()
    public static final int ROUND_CEILING;

    @Deprecated()
    public static final int ROUND_FLOOR;

    @Deprecated()
    public static final int ROUND_HALF_UP;

    @Deprecated()
    public static final int ROUND_HALF_DOWN;

    @Deprecated()
    public static final int ROUND_HALF_EVEN;

    @Deprecated()
    public static final int ROUND_UNNECESSARY;

    public BigDecimal round(MathContext mc);

    public BigDecimal setScale(int newScale, RoundingMode roundingMode);

    @Deprecated()
    public BigDecimal setScale(int newScale, int roundingMode);

    public BigDecimal setScale(int newScale);

    public BigDecimal movePointLeft(int n);

    public BigDecimal movePointRight(int n);

    public BigDecimal scaleByPowerOfTen(int n);

    public BigDecimal stripTrailingZeros();

    @Override
    public int compareTo(BigDecimal val);

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object x);

    @Pure
    @StaticallyExecutable
    public BigDecimal min(BigDecimal val);

    @Pure
    @StaticallyExecutable
    public BigDecimal max(BigDecimal val);

    @Override
    public int hashCode();

    @Override
    public String toString();

    public String toEngineeringString();

    public String toPlainString();

    public BigInteger toBigInteger();

    public BigInteger toBigIntegerExact();

    @Override
    @PolyValue
    public long longValue(@PolyValue BigDecimal this);

    public long longValueExact();

    private static class LongOverflow {

        public static void check(BigDecimal num);
    }

    @Override
    @PolyValue
    public int intValue(@PolyValue BigDecimal this);

    public int intValueExact();

    public short shortValueExact();

    public byte byteValueExact();

    @Override
    @PolyValue
    public float floatValue(@PolyValue BigDecimal this);

    @Override
    @PolyValue
    public double doubleValue(@PolyValue BigDecimal this);

    public BigDecimal ulp();

    static class StringBuilderHelper {

        StringBuilder getStringBuilder();

        char[] getCompactCharArray();

        int putIntCompact(long intCompact);
    }

    private static class UnsafeHolder {

        static void setIntCompact(BigDecimal bd, long val);

        static void setIntValVolatile(BigDecimal bd, BigInteger val);
    }

    static int longDigitLength(long x);

    static BigDecimal scaledTenPow(int n, int sign, int scale);
}
