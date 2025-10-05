/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.PolyLowerBound;
    @Positive
import org.checkerframework.checker.index.qual.PolyUpperBound;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.common.value.qual.StaticallyExecutable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.util.Random;
    @Positive
import jdk.internal.math.FloatConsts;
    @Positive
import jdk.internal.math.DoubleConsts;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;

    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Math {

    @Positive
    public static final double E;

    @Positive
    public static final double PI;

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double sin(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double cos(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double tan(double a);

    @Positive
    @Pure
    @Positive
    public static double asin(double a);

    @Positive
    @Pure
    @Positive
    public static double acos(double a);

    @Positive
    @Pure
    @Positive
    public static double atan(double a);

    @Positive
    @Pure
    @Positive
    public static double toRadians(double angdeg);

    @Positive
    @Pure
    @Positive
    public static double toDegrees(double angrad);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double exp(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double log(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double log10(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double sqrt(double a);

    @Positive
    @Pure
    @Positive
    public static double cbrt(double a);

    @Positive
    @Pure
    @Positive
    public static double IEEEremainder(double f1, double f2);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double ceil(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double floor(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double rint(double a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double atan2(double y, double x);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double pow(double a, double b);

    @Positive
    @Pure
    @Positive
    public static int round(float a);

    @Positive
    @Pure
    @Positive
    public static long round(double a);

    @Positive
    private static final class RandomNumberGeneratorHolder {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public static double random();

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int addExact(int x, int y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long addExact(long x, long y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int subtractExact(int x, int y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long subtractExact(long x, long y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int multiplyExact(int x, int y);

    @Positive
    public static long multiplyExact(long x, int y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long multiplyExact(long x, long y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int incrementExact(int a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long incrementExact(long a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int decrementExact(int a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long decrementExact(long a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static int negateExact(int a);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long negateExact(long a);

    @Positive
    public static int toIntExact(long value);

    @Positive
    public static long multiplyFull(int x, int y);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static long multiplyHigh(long x, long y);

    @Positive
    public static int floorDiv(int x, int y);

    @Positive
    public static long floorDiv(long x, int y);

    @Positive
    public static long floorDiv(long x, long y);

    @Positive
    public static int floorMod(int x, int y);

    @Positive
    public static int floorMod(long x, int y);

    @Positive
    public static long floorMod(long x, long y);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    public static int abs(int a);

    @Positive
    @Pure
    @Positive
    public static int absExact(int a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @NonNegative
    @Positive
    public static long abs(long a);

    @Positive
    @Pure
    @Positive
    public static long absExact(long a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float abs(float a);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double abs(double a);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyUpperBound
    @Positive
    public static int max(@PolyUpperBound int a, @PolyUpperBound int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyUpperBound
    @Positive
    public static long max(@PolyUpperBound long a, @PolyUpperBound long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float max(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double max(double a, double b);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyLowerBound
    @Positive
    public static int min(@PolyLowerBound int a, @PolyLowerBound int b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @PolyLowerBound
    @Positive
    public static long min(@PolyLowerBound long a, @PolyLowerBound long b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float min(float a, float b);

    @Positive
    @Pure
    @Positive
    @StaticallyExecutable
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double min(double a, double b);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static double fma(double a, double b, double c);

    @Positive
    @IntrinsicCandidate
    @Positive
    public static float fma(float a, float b, float c);

    @Positive
    @Pure
    @Positive
    public static double ulp(double d);

    @Positive
    @Pure
    @Positive
    public static float ulp(float f);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double signum(double d);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float signum(float f);

    @Positive
    @Pure
    @Positive
    public static double sinh(double x);

    @Positive
    @Pure
    @Positive
    public static double cosh(double x);

    @Positive
    @Pure
    @Positive
    public static double tanh(double x);

    @Positive
    @Pure
    @Positive
    public static double hypot(double x, double y);

    @Positive
    @Pure
    @Positive
    public static double expm1(double x);

    @Positive
    @Pure
    @Positive
    public static double log1p(double x);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static double copySign(double magnitude, double sign);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public static float copySign(float magnitude, float sign);

    @Positive
    @Pure
    @Positive
    public static int getExponent(float f);

    @Positive
    @Pure
    @Positive
    public static int getExponent(double d);

    @Positive
    @Pure
    @Positive
    public static double nextAfter(double start, double direction);

    @Positive
    @Pure
    @Positive
    public static float nextAfter(float start, double direction);

    @Positive
    @Pure
    @Positive
    public static double nextUp(double d);

    @Positive
    @Pure
    @Positive
    public static float nextUp(float f);

    @Positive
    public static double nextDown(double d);

    @Positive
    public static float nextDown(float f);

    @Positive
    @Pure
    @Positive
    public static double scalb(double d, int scaleFactor);

    @Positive
    @Pure
    @Positive
    public static float scalb(float f, int scaleFactor);

    @Positive
    static double powerOfTwoD(int n);

    @Positive
    static float powerOfTwoF(int n);
    @Positive
}
