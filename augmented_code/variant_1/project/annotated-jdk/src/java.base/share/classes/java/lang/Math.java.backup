/*
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.PolyLowerBound;
import org.checkerframework.checker.index.qual.PolyUpperBound;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.common.value.qual.StaticallyExecutable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.math.BigDecimal;
import java.util.Random;
import jdk.internal.math.FloatConsts;
import jdk.internal.math.DoubleConsts;
import jdk.internal.vm.annotation.IntrinsicCandidate;

@AnnotatedFor({ "index", "interning", "lock", "nullness" })
@UsesObjectEquals
public final class Math {

    public static final double E;

    public static final double PI;

    @Pure
    @IntrinsicCandidate
    public static double sin(double a);

    @Pure
    @IntrinsicCandidate
    public static double cos(double a);

    @Pure
    @IntrinsicCandidate
    public static double tan(double a);

    @Pure
    public static double asin(double a);

    @Pure
    public static double acos(double a);

    @Pure
    public static double atan(double a);

    @Pure
    public static double toRadians(double angdeg);

    @Pure
    public static double toDegrees(double angrad);

    @Pure
    @IntrinsicCandidate
    public static double exp(double a);

    @Pure
    @IntrinsicCandidate
    public static double log(double a);

    @Pure
    @IntrinsicCandidate
    public static double log10(double a);

    @Pure
    @IntrinsicCandidate
    public static double sqrt(double a);

    @Pure
    public static double cbrt(double a);

    @Pure
    public static double IEEEremainder(double f1, double f2);

    @Pure
    @IntrinsicCandidate
    public static double ceil(double a);

    @Pure
    @IntrinsicCandidate
    public static double floor(double a);

    @Pure
    @IntrinsicCandidate
    public static double rint(double a);

    @Pure
    @IntrinsicCandidate
    public static double atan2(double y, double x);

    @Pure
    @IntrinsicCandidate
    public static double pow(double a, double b);

    @Pure
    public static int round(float a);

    @Pure
    public static long round(double a);

    private static final class RandomNumberGeneratorHolder {
    }

    @Pure
    public static double random();

    @IntrinsicCandidate
    public static int addExact(int x, int y);

    @IntrinsicCandidate
    public static long addExact(long x, long y);

    @IntrinsicCandidate
    public static int subtractExact(int x, int y);

    @IntrinsicCandidate
    public static long subtractExact(long x, long y);

    @IntrinsicCandidate
    public static int multiplyExact(int x, int y);

    public static long multiplyExact(long x, int y);

    @IntrinsicCandidate
    public static long multiplyExact(long x, long y);

    @IntrinsicCandidate
    public static int incrementExact(int a);

    @IntrinsicCandidate
    public static long incrementExact(long a);

    @IntrinsicCandidate
    public static int decrementExact(int a);

    @IntrinsicCandidate
    public static long decrementExact(long a);

    @IntrinsicCandidate
    public static int negateExact(int a);

    @IntrinsicCandidate
    public static long negateExact(long a);

    public static int toIntExact(long value);

    public static long multiplyFull(int x, int y);

    @IntrinsicCandidate
    public static long multiplyHigh(long x, long y);

    public static int floorDiv(int x, int y);

    public static long floorDiv(long x, int y);

    public static long floorDiv(long x, long y);

    public static int floorMod(int x, int y);

    public static int floorMod(long x, int y);

    public static long floorMod(long x, long y);

    @Pure
    @IntrinsicCandidate
    @NonNegative
    public static int abs(int a);

    @Pure
    public static int absExact(int a);

    @Pure
    @IntrinsicCandidate
    @NonNegative
    public static long abs(long a);

    @Pure
    public static long absExact(long a);

    @Pure
    @IntrinsicCandidate
    public static float abs(float a);

    @Pure
    @IntrinsicCandidate
    public static double abs(double a);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    @PolyUpperBound
    public static int max(@PolyUpperBound int a, @PolyUpperBound int b);

    @Pure
    @StaticallyExecutable
    @PolyUpperBound
    public static long max(@PolyUpperBound long a, @PolyUpperBound long b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static float max(float a, float b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static double max(double a, double b);

    @Pure
    @IntrinsicCandidate
    @PolyLowerBound
    public static int min(@PolyLowerBound int a, @PolyLowerBound int b);

    @Pure
    @StaticallyExecutable
    @PolyLowerBound
    public static long min(@PolyLowerBound long a, @PolyLowerBound long b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static float min(float a, float b);

    @Pure
    @StaticallyExecutable
    @IntrinsicCandidate
    public static double min(double a, double b);

    @IntrinsicCandidate
    public static double fma(double a, double b, double c);

    @IntrinsicCandidate
    public static float fma(float a, float b, float c);

    @Pure
    public static double ulp(double d);

    @Pure
    public static float ulp(float f);

    @Pure
    @IntrinsicCandidate
    public static double signum(double d);

    @Pure
    @IntrinsicCandidate
    public static float signum(float f);

    @Pure
    public static double sinh(double x);

    @Pure
    public static double cosh(double x);

    @Pure
    public static double tanh(double x);

    @Pure
    public static double hypot(double x, double y);

    @Pure
    public static double expm1(double x);

    @Pure
    public static double log1p(double x);

    @Pure
    @IntrinsicCandidate
    public static double copySign(double magnitude, double sign);

    @Pure
    @IntrinsicCandidate
    public static float copySign(float magnitude, float sign);

    @Pure
    public static int getExponent(float f);

    @Pure
    public static int getExponent(double d);

    @Pure
    public static double nextAfter(double start, double direction);

    @Pure
    public static float nextAfter(float start, double direction);

    @Pure
    public static double nextUp(double d);

    @Pure
    public static float nextUp(float f);

    public static double nextDown(double d);

    public static float nextDown(float f);

    @Pure
    public static double scalb(double d, int scaleFactor);

    @Pure
    public static float scalb(float f, int scaleFactor);

    static double powerOfTwoD(int n);

    static float powerOfTwoF(int n);
}
